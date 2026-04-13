# LLVM Code Map for LLVM2 AArch64 Backend

**Author:** Andrew Yates <ayates@dropbox.com>
**Date:** 2026-04-12
**Source:** `~/llvm-project-ref/` (LLVM trunk)
**Purpose:** File:line reference for porting LLVM's AArch64 backend algorithms to Rust for LLVM2

---

## 1. AArch64 Backend (`llvm/lib/Target/AArch64/`)

### 1.1 Instruction Selection Lowering

**File:** `AArch64ISelLowering.cpp` (33,715 lines)

This is the largest and most important file in the backend. It defines how LLVM IR
operations are lowered to AArch64-specific SelectionDAG nodes.

| Function | Line | Purpose |
|----------|------|---------|
| `AArch64TargetLowering::AArch64TargetLowering()` | 412 | **Constructor** — registers legal types, sets up operation actions, defines which operations need custom lowering. This is the single most important function for understanding what the backend supports. |
| `LowerOperation()` | 8190 | **Main dispatch** — routes each ISD opcode to its specific Lower* handler. The central switch statement for all custom-lowered operations. |
| `LowerFormalArguments()` | 8889 | Lowers function arguments per calling convention. Handles register args, stack args, byval, sret. |
| `LowerCall()` | 10045 | Lowers function calls. Handles argument passing, stack setup, call instruction emission, return value extraction. |
| `LowerCallResult()` | 9503 | Extracts return values from physical registers after a call. |
| `LowerReturn()` | 10805 | Lowers return statements. Copies return values to physical registers per CC. |
| `isEligibleForTailCallOptimization()` | 9669 | Decides whether a call can be tail-call optimized. |
| `PerformDAGCombine()` | 29523 | **Target-specific DAG combines** — peephole optimizations on the SelectionDAG before instruction selection. Massive function with many sub-combines. |

**Key lowering functions by category:**

*Arithmetic:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerMUL()` | 6015 | Multiply lowering, including widening multiplies |
| `LowerDIV()` | 17141 | Division lowering |
| `LowerXOR()` | 4541 | XOR with special-case optimizations |
| `LowerABS()` | 7857 | Absolute value lowering |
| `LowerFMUL()` | 8049 | Floating-point multiply |
| `LowerFMA()` | 8121 | Fused multiply-add |

*Floating-point:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerFP_EXTEND()` | 4796 | FP widening conversions |
| `LowerFP_ROUND()` | 4841 | FP narrowing conversions |
| `LowerFP_TO_INT()` | 5086 | FP to integer conversion |
| `LowerFP_TO_INT_SAT()` | 5222 | Saturating FP to integer |
| `LowerINT_TO_FP()` | 5430 | Integer to FP conversion |
| `LowerFCOPYSIGN()` | 11830 | Copy sign bit between FP values |
| `LowerGET_ROUNDING()` | 5765 | Read FP rounding mode |
| `LowerSET_ROUNDING()` | 5789 | Set FP rounding mode |

*Control flow:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerBR_CC()` | 11661 | Conditional branch lowering |
| `LowerSETCC()` | 12242 | Condition code comparison |
| `LowerSELECT()` | 12813 | Conditional select (CSEL) |
| `LowerSELECT_CC()` | 12801 | Select with embedded compare |
| `LowerBR_JT()` | 12908 | Jump table branch lowering |
| `LowerJumpTable()` | 12893 | Jump table address materialization |
| `LowerBRIND()` | 12951 | Indirect branch |

*Memory:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerLOAD()` | 7772 | Custom load lowering |
| `LowerSTORE()` | 7612 | Custom store lowering |
| `LowerStore128()` | 7670 | 128-bit store (STP) lowering |
| `LowerMLOAD()` | 7427 | Masked load |
| `LowerMGATHER()` | 7245 | Masked gather |
| `LowerMSCATTER()` | 7345 | Masked scatter |

*Address materialization:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerGlobalAddress()` | 11045 | Global variable address (ADRP+ADD or GOT) |
| `LowerGlobalTLSAddress()` | 11470 | Thread-local storage address |
| `LowerBlockAddress()` | 12995 | Basic block address |
| `LowerConstantPool()` | 12978 | Constant pool entry address |

*Varargs:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerVASTART()` | 13154 | va_start lowering (dispatches to platform-specific) |
| `LowerDarwin_VASTART()` | 13030 | Darwin va_start |
| `LowerAAPCS_VASTART()` | 13075 | AAPCS va_start (separate GPR/FPR save areas) |
| `LowerVACOPY()` | 13167 | va_copy lowering |
| `LowerVAARG()` | 13187 | va_arg lowering |

*Vector operations:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerBUILD_VECTOR()` | 16319 | Vector constant/shuffle construction |
| `LowerVECTOR_SHUFFLE()` | 15328 | Vector permutation |
| `LowerCONCAT_VECTORS()` | 16828 | Vector concatenation |
| `LowerINSERT_VECTOR_ELT()` | 16865 | Element insertion |
| `LowerEXTRACT_SUBVECTOR()` | 16960 | Subvector extraction |
| `LowerINSERT_SUBVECTOR()` | 17014 | Subvector insertion |
| `LowerSPLAT_VECTOR()` | 15531 | Vector splat |
| `LowerVECTOR_SPLICE()` | 12759 | SVE vector splice |
| `LowerVectorOR()` | 16076 | Optimized vector OR patterns |
| `LowerVECTOR_COMPRESS()` | 7818 | SVE compress |

*Bit manipulation:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerCTPOP_PARITY()` | 11940 | Population count / parity |
| `LowerCTTZ()` | 12069 | Count trailing zeros |
| `LowerBitreverse()` | 12128 | Bit reversal |
| `LowerMinMax()` | 12080 | Min/max operations |
| `LowerShiftParts()` | 13361 | 128-bit shift lowering |

*Miscellaneous:*
| Function | Line | Purpose |
|----------|------|---------|
| `LowerBITCAST()` | 5617 | Bitcast between types |
| `LowerFRAMEADDR()` | 13258 | Frame address intrinsic |
| `LowerRETURNADDR()` | 13320 | Return address intrinsic |
| `LowerINTRINSIC_VOID()` | 6422 | Void-returning intrinsics |
| `LowerINTRINSIC_W_CHAIN()` | 6480 | Chained intrinsics |
| `LowerINTRINSIC_WO_CHAIN()` | 6513 | Pure intrinsics |

---

### 1.2 Instruction Selection (DAG to DAG)

**File:** `AArch64ISelDAGToDAG.cpp` (8,009 lines)

Pattern-matching from generic SelectionDAG nodes to AArch64 machine instructions.

| Function | Line | Purpose |
|----------|------|---------|
| `Select()` | 4958 | **Main entry point** — the giant switch statement that matches DAG nodes to machine instructions. |
| `SelectArithImmed()` | 715 | Match arithmetic immediate operands (12-bit with optional shift) |
| `SelectNegArithImmed()` | 745 | Match negated arithmetic immediates |
| `SelectShiftedRegister()` | 1009 | Match shifted register operands (LSL/LSR/ASR/ROR) |
| `SelectShiftedRegisterFromAnd()` | 853 | Match shifted registers from AND patterns |
| `SelectArithExtendedRegister()` | 1080 | Match extended register operands (SXTB/UXTW etc.) |
| `SelectAddrModeIndexed()` | 1263 | Match base+offset addressing modes |
| `SelectAddrModeUnscaled()` | 1323 | Match unscaled (LDUR/STUR) addressing |
| `SelectAddrModeWRO()` | 1387 | Match register-offset addressing (W-reg) |
| `SelectAddrModeXRO()` | 1476 | Match register-offset addressing (X-reg) |
| `SelectExtendedSHL()` | 1355 | Match extended shift-left patterns |
| `SelectTable()` | 1620 | Select TBL/TBX vector table lookup |
| `SelectLoad()` | 1894 | Select multi-vector loads (LD2/LD3/LD4) |

---

### 1.3 Frame Lowering (Prologue/Epilogue)

**File:** `AArch64FrameLowering.cpp` (4,104 lines)

Handles stack frame layout, callee-saved register spilling, and prologue/epilogue emission.

| Function | Line | Purpose |
|----------|------|---------|
| `emitPrologue()` | 1214 | Delegates to `AArch64PrologueEmitter` |
| `emitEpilogue()` | 1220 | Delegates to `AArch64EpilogueEmitter` |
| `canUseRedZone()` | 533 | Determines if leaf function can use red zone (skip SP adjustment) |
| `hasFPImpl()` | 563 | Determines if frame pointer is needed |
| `isFPReserved()` | 619 | Whether FP register (X29) is reserved |
| `hasReservedCallFrame()` | 644 | Whether call frame pseudo instructions can be eliminated |
| `eliminateCallFramePseudoInstr()` | 654 | Replaces ADJCALLSTACKDOWN/UP with SP adjustments |
| `resolveFrameIndexReference()` | 1350 | Resolves abstract frame index to concrete SP/FP offset |
| `resolveFrameOffsetReference()` | 1361 | Computes actual offset from frame register |
| `getFPOffset()` | 1319 | Computes FP-relative offset for a frame index |
| `spillCalleeSavedRegisters()` | 1949 | Emits STP instructions to save callee-saved regs |
| `restoreCalleeSavedRegisters()` | 2168 | Emits LDP instructions to restore callee-saved regs |
| `determineCalleeSaves()` | 2495 | Decides which registers must be saved |
| `assignCalleeSavedSpillSlots()` | 2766 | Assigns stack slots for callee-saved regs |
| `processFunctionBeforeFrameFinalized()` | 2987 | Final frame adjustments before frame index elimination |
| `orderFrameObjects()` | 3647 | Orders stack objects for optimal access |
| `inlineStackProbe()` | 3910 | Emits inline stack probes for large frames |
| `inlineStackProbeFixed()` | 3829 | Fixed-size stack probing |
| `emitRemarks()` | 4001 | Emits optimization remarks about frame decisions |
| `producePairRegisters()` | 449 | Whether to pair callee-saved registers in STP/LDP |
| `resetCFIToInitialState()` | 717 | Resets CFI state for async unwinding |
| `emitZeroCallUsedRegs()` | 827 | Zeros caller-saved registers for security |

**File:** `AArch64PrologueEpilogue.cpp` (1,798 lines)

The actual prologue/epilogue emission logic (separated from FrameLowering).

| Function | Line | Purpose |
|----------|------|---------|
| `AArch64PrologueEmitter::emitPrologue()` | 638 | **Main prologue emission** — SP adjustment, FP setup, callee-save spills, CFI directives |
| `AArch64EpilogueEmitter::emitEpilogue()` | 1355 | **Main epilogue emission** — restore callee-saves, SP restore, return |

---

### 1.4 Binary Encoding (MC Code Emitter)

**File:** `MCTargetDesc/AArch64MCCodeEmitter.cpp` (784 lines)

Encodes `MCInst` objects to binary machine code bytes.

| Function | Line | Purpose |
|----------|------|---------|
| `encodeInstruction()` | 725 | **Main entry** — encodes MCInst to bytes. Most encoding is TableGen-generated; this handles special cases (TLSDESCCALL, SPACE) and emits 4-byte little-endian encoding. |
| `getCondBranchTargetOpValue()` | 323 | Encodes conditional branch target (19-bit PC-relative) |
| `getCondCompBranchTargetOpValue()` | 341 | Encodes compare-and-branch target |
| `getTestBranchTargetOpValue()` | 425 | Encodes test-and-branch target (14-bit offset) |
| `getFixedPointScaleOpValue()` | 496 | Encodes fixed-point scale factor |
| `getMoveVecShifterOpValue()` | 682 | Encodes vector move shifter |
| `fixMOVZ()` | 693 | Applies MOVZ fixup for immediate materialization |
| `fixOneOperandFPComparison()` | 771 | Fixes FCMP with zero encoding |
| `EncodeZK()` | 591 | Encodes SVE Z-register with index |
| `EncodeZPR2StridedRegisterClass()` | 615 | Encodes SVE strided register pair |
| `EncodeZPR4StridedRegisterClass()` | 625 | Encodes SVE strided register quad |
| `EncodeMatrixTileListRegisterClass()` | 635 | Encodes SME matrix tile list |

---

### 1.5 Mach-O Object Writer (AArch64)

**File:** `MCTargetDesc/AArch64MachObjectWriter.cpp` (437 lines)

Handles AArch64-specific Mach-O relocations.

| Function | Line | Purpose |
|----------|------|---------|
| `getAArch64FixupKindMachOInfo()` | 50 | **Maps fixup kinds to Mach-O relocation types.** Central mapping table: `fixup_aarch64_pcrel_adrp_imm21` -> `ARM64_RELOC_PAGE21`, `fixup_aarch64_add_imm12` -> `ARM64_RELOC_PAGEOFF12`, etc. |
| `recordRelocation()` | 148 | **Records a relocation** for the Mach-O object file. Handles paired relocations (ADRP+ADD, ADRP+LDR), subtractor relocations, GOT references, TLV references. |

---

### 1.6 AArch64 Fixup Kinds

**File:** `MCTargetDesc/AArch64FixupKinds.h` (74 lines)

Defines all target-specific fixup types. These directly map to relocation types.

| Fixup | Line | Purpose |
|-------|------|---------|
| `fixup_aarch64_pcrel_adr_imm21` | 19 | ADR instruction — 21-bit PC-relative |
| `fixup_aarch64_pcrel_adrp_imm21` | 22 | ADRP instruction — 21-bit page-relative |
| `fixup_aarch64_add_imm12` | 26 | ADD/SUB 12-bit immediate |
| `fixup_aarch64_ldst_imm12_scale{1,2,4,8,16}` | 29-33 | Load/store unsigned 12-bit offset (scaled by element size) |
| `fixup_aarch64_ldr_pcrel_imm19` | 38 | PC-relative load (LDR literal) — 19-bit |
| `fixup_aarch64_movw` | 41 | MOVZ/MOVK/MOVN immediate |
| `fixup_aarch64_pcrel_branch9` | 44 | 9-bit branch (Bcc.cond) |
| `fixup_aarch64_pcrel_branch14` | 47 | 14-bit branch (TBZ/TBNZ) |
| `fixup_aarch64_pcrel_branch16` | 52 | 16-bit branch (pointer auth) |
| `fixup_aarch64_pcrel_branch19` | 57 | 19-bit branch (B.cond, CBZ/CBNZ) |
| `fixup_aarch64_pcrel_branch26` | 60 | 26-bit branch (B) |
| `fixup_aarch64_pcrel_call26` | 64 | 26-bit call (BL) |

---

### 1.7 AArch64 AsmBackend (Fixup Application + Compact Unwind)

**File:** `MCTargetDesc/AArch64AsmBackend.cpp`

| Function | Line | Purpose |
|----------|------|---------|
| `getFixupKindInfo()` | 44 | Returns name, offset, size, flags for each fixup kind |
| `shouldForceRelocation()` | 404 | Determines if fixup must become a relocation (vs resolved at assembly time) |
| `applyFixup()` | 420 | **Applies a fixup value to encoded instruction bytes.** Handles bit-field insertion for each fixup kind. |
| `generateCompactUnwindEncoding()` | 576 | **Generates Darwin compact unwind encoding** from CFI directives. Encodes frame layout (FP-based or frameless), saved register pairs, stack size. |

**Compact unwind encoding constants** (line 517):
| Constant | Value | Meaning |
|----------|-------|---------|
| `UNWIND_ARM64_MODE_FRAMELESS` | `0x02000000` | Leaf function, no frame pointer |
| `UNWIND_ARM64_MODE_DWARF` | `0x03000000` | Fallback to DWARF FDE |
| `UNWIND_ARM64_MODE_FRAME` | `0x04000000` | Standard FP/LR frame |
| `UNWIND_ARM64_FRAME_X19_X20_PAIR` | `0x00000001` | X19/X20 saved |
| `UNWIND_ARM64_FRAME_X21_X22_PAIR` | `0x00000002` | X21/X22 saved |
| `UNWIND_ARM64_FRAME_X23_X24_PAIR` | `0x00000004` | X23/X24 saved |
| `UNWIND_ARM64_FRAME_X25_X26_PAIR` | `0x00000008` | X25/X26 saved |
| `UNWIND_ARM64_FRAME_X27_X28_PAIR` | `0x00000010` | X27/X28 saved |
| `UNWIND_ARM64_FRAME_D8_D9_PAIR` | `0x00000100` | D8/D9 saved |
| `UNWIND_ARM64_FRAME_D10_D11_PAIR` | `0x00000200` | D10/D11 saved |
| `UNWIND_ARM64_FRAME_D12_D13_PAIR` | `0x00000400` | D12/D13 saved |
| `UNWIND_ARM64_FRAME_D14_D15_PAIR` | `0x00000800` | D14/D15 saved |

Frameless encoding uses bits [12:23] for stack size (`stack_size / 16`), computed by `encodeStackAdjustment()` (line 558).

---

### 1.8 Immediate Materialization

**File:** `AArch64ExpandImm.cpp` (722 lines)

Synthesizes arbitrary 64-bit immediates using minimal instruction sequences.

| Function | Line | Purpose |
|----------|------|---------|
| `expandMOVImm()` | 598 | **Main entry point.** Given a 64-bit immediate and bit size (32/64), produces a sequence of MOV/ORR/MOVK instructions. Strategy cascade: (1) single MOVZ/MOVN if most chunks are 0/1, (2) single ORR if logical immediate, (3) MOVZ+MOVK pairs, (4) ORR+MOVK, (5) two ORR instructions, (6) ORR+AND, (7) ORR+EOR, (8) MOVN+EOR, (9) three-instruction fallbacks, (10) four MOVZ/MOVK. |
| `tryToreplicateChunks()` | 43 | Finds repeated 16-bit chunks and uses ORR+MOVK |
| `trySequenceOfOnes()` | 150 | Finds contiguous 1-bit sequences for ORR+MOVK |
| `getChunk()` | 22 | Extracts 16-bit chunk from 64-bit value |
| `canUseOrr()` | 30 | Tests if replicated chunk is a valid ORR-immediate |
| `isStartChunk()` | 106 | Pattern match for `1...0...` bit pattern |
| `isEndChunk()` | 116 | Pattern match for `0...1...` bit pattern |

**Key insight for porting:** The AArch64 logical immediate encoding is a bitmask that encodes `(element_size, num_ones, rotation)`. The function `processLogicalImmediate()` (in `AArch64AddressingModes.h`) performs this encoding and is essential for both immediate materialization and instruction encoding. It is one of the trickiest parts of the backend.

---

### 1.9 Calling Convention

**File:** `AArch64CallingConvention.td`

TableGen definitions for argument/return value assignment to registers/stack.

| Definition | Line | Purpose |
|------------|------|---------|
| `AArch64_Common` | 30 | Shared rules: nest in X15, sret in X8, SwiftSelf in X20, SwiftError in X21 |
| `CC_AArch64_AAPCS` | 126 | **AAPCS64 calling convention** — i32 in W0-W7, i64 in X0-X7, f32 in S0-S7, f64 in D0-D7, vectors in Q0-Q7, SVE in Z0-Z7. Stack slots padded to 8 bytes. |
| `RetCC_AArch64_AAPCS` | 129 | Return value convention for AAPCS64 |
| `CC_AArch64_DarwinPCS` | 366 | **Darwin/Apple calling convention** — differs from AAPCS: i128 splits don't require even register pairs; stack slots sized to actual type (not padded to 8 bytes). |
| `CC_AArch64_DarwinPCS_VarArg` | 442 | Darwin variadic function convention — all varargs on stack |
| `CC_AArch64_Win64PCS` | 165 | Windows ARM64 calling convention |
| `CC_AArch64_Win64_VarArg` | 175 | Windows ARM64 variadic convention |
| `CC_AArch64_Arm64EC_VarArg` | 185 | Arm64EC (emulation compatible) variadic |

**Key differences Darwin vs AAPCS (line 361-364):**
1. i128 (split i64 pairs) do NOT need even-aligned register pairs on Darwin
2. Stack slots are naturally sized (1/2/4/8 bytes) on Darwin vs always 8-byte on AAPCS

---

### 1.10 Register File

**File:** `AArch64RegisterInfo.td` (2,014 lines)

| Definition | Line | Purpose |
|------------|------|---------|
| `AArch64Reg` | 13 | Base register class with hardware encoding |
| SubRegIndex definitions | 22-76 | Sub-register hierarchy (sub_32, bsub, hsub, ssub, dsub, zsub, etc.) |
| `GPR32` | 202 | 32-bit GPR class (W0-W30 + WZR) |
| `GPR64` | 207 | 64-bit GPR class (X0-X30 + XZR) |
| `GPR32sp` / `GPR64sp` | 214/219 | GPR classes including SP |
| `GPR64noip` | 282 | GPRs excluding X16, X17, LR (for pointer auth) |
| `FPR8` / `FPR16` / `FPR32` / `FPR64` / `FPR128` | 501-538 | NEON/FP register classes at each width |
| `XSeqPairs` / `WSeqPairs` | 804/801 | Register pair tuples for i128 |
| `PPR` / `PPR_3b` | 989/992 | SVE predicate registers (P0-P15) |
| `ZPR` / `ZPR_4b` / `ZPR_3b` | 1192/1195/1198 | SVE vector registers (Z0-Z31) |
| `ZPR2` / `ZPR3` / `ZPR4` | 1303/1307/1311 | SVE register tuples |

---

### 1.11 Instruction Formats

**File:** `AArch64InstrFormats.td` (very large)

| Definition | Line | Purpose |
|------------|------|---------|
| `AArch64Inst` | 79 | **Base instruction class** — 32-bit encoding, format field |
| `Format` | 30 | Encoding format (PseudoFrm=0, NormalFrm=1) |
| `EncodedI` | 126 | Base for encoded instructions with pattern |
| `I` | 132 | Standard instruction template (outputs, inputs, asm, pattern) |
| `DestructiveInstTypeEnum` | 39 | SVE destructive instruction classification |
| `BaseSystemI` | 1810 | System instruction base |
| `ADRI` | 2915 | ADR/ADRP instruction format |
| `BaseLoadStoreUI` | 3836 | Load/store unsigned immediate format |
| `BaseCASEncoding` | 12379 | Compare-and-swap encoding |

---

### 1.12 Instruction Info (Target Hooks)

**File:** `AArch64InstrInfo.cpp` (11,838 lines)

| Function | Line | Purpose |
|----------|------|---------|
| `getInstSizeInBytes()` | 112 | Returns size of instruction in bytes (4 for most, variable for pseudos) |
| `analyzeBranch()` | 407 | **Analyzes branch structure** of a basic block. Returns fall-through, conditional, and unconditional branch targets. Critical for block layout. |
| `analyzeBranchPredicate()` | 523 | Extracts branch predicate for if-conversion |
| `reverseBranchCondition()` | 602 | Inverts a branch condition |
| `removeBranch()` | 656 | Removes branch instructions from block end |
| `insertBranch()` | 723 | Inserts branch instructions at block end |
| `isCoalescableExtInstr()` | 1442 | Identifies extension instructions that can be coalesced |
| `isSchedulingBoundary()` | 1501 | Identifies instructions that block scheduling |
| `copyPhysReg()` | 5502 | Emits register-to-register copy (MOV, FMOV, etc.) |
| `copyPhysRegTuple()` | 5418 | Copies register tuples (for SVE, NEON multi-reg) |
| `storeRegToStackSlot()` | 6054 | Emits store to spill slot |
| `loadRegFromStackSlot()` | 6234 | Emits load from spill slot |

---

### 1.13 Load/Store Optimizer

**File:** `AArch64LoadStoreOptimizer.cpp` (3,179 lines)

Post-regalloc pass that merges adjacent loads/stores into LDP/STP.

| Function | Line | Purpose |
|----------|------|---------|
| `runOnMachineFunction()` | 3120 | Entry point |
| `optimizeBlock()` | 3015 | Per-block optimization loop |
| `tryToPromoteLoadFromStore()` | 2786 | Forwards store data to subsequent load |
| `tryToMergeZeroStInst()` | 2815 | Merges stores of zero (STP XZR, XZR) |
| `tryToPairLdStInst()` | 2841 | **Core pairing logic** — finds adjacent loads/stores and merges to LDP/STP |
| `tryToMergeLdStUpdate()` | 2920 | Merges base-update into pre/post-indexed form |
| `tryToMergeIndexLdSt()` | 2987 | Merges indexed load/store patterns |
| `findMatchingStore()` | 1605 | Searches for store that matches a load for forwarding |
| `isMatchingUpdateInsn()` | 2451 | Checks if an instruction can be folded as base update |
| `isMatchingMovConstInsn()` | 2498 | Checks if MOV const can be merged with store |

---

## 2. CodeGen Infrastructure (`llvm/lib/CodeGen/`)

### 2.1 Register Allocation (Greedy)

**File:** `RegAllocGreedy.cpp` + `RegAllocGreedy.h`

The default register allocator. Assigns virtual registers to physical registers using a priority-based greedy approach with splitting and eviction.

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `run()` | .cpp | 2931 | Entry point — initializes data structures and runs allocation loop |
| `selectOrSplit()` | .cpp | 2322 | **Top-level allocation** — try assign, then split, then last-chance recoloring |
| `selectOrSplitImpl()` | .cpp | 2644 | Implementation of the allocation decision cascade |
| `tryAssign()` | .cpp | 535 | Attempts direct register assignment from allocation order |
| `tryEvict()` | .cpp | 716 | Evicts a lower-priority live range to make room |
| `trySplit()` | .cpp | 1969 | Splits a live range at region boundaries |
| `tryLastChanceRecoloring()` | .cpp | 2126 | Last resort: recolor interfering assignments |
| `tryAssignCSRFirstTime()` | .cpp | 2378 | Special handling for callee-saved registers |
| `trySplitAroundHintReg()` | .cpp | 1370 | Splits live range around a hinted register |
| `evictInterference()` | .cpp | 620 | Performs eviction of interfering ranges |
| `splitAroundRegion()` | .cpp | 1061 | Region-based live range splitting |
| `growRegion()` | .cpp | 866 | Grows split region to find optimal split points |
| `calcCompactRegion()` | .cpp | 945 | Computes compact region for splitting |
| `calculateRegionSplitCost()` | .cpp | 1310 | Costs a potential region split |
| `enqueue()` | .cpp | 423 | Adds live interval to priority queue |
| `calcGapWeights()` | .cpp | 1660 | Computes gap weights for local splitting |
| `tryHintRecoloring()` | .cpp | 2523 | Tries to recolor to honor register hints |
| `tryHintsRecoloring()` | .cpp | 2632 | Batch hint recoloring |
| `mayRecolorAllInterferences()` | .cpp | 2039 | Checks if all interferences can be recolored |
| `tryRecoloringCandidates()` | .cpp | 2286 | Attempts recoloring of candidate set |
| `initializeCSRCost()` | .cpp | 2417 | Sets up callee-saved register cost model |

**Algorithm overview:** The allocator processes live intervals in priority order (longest/most-connected first). For each interval, it tries: (1) direct assignment, (2) eviction of lower-priority intervals, (3) region splitting, (4) last-chance recoloring. Splitting creates sub-intervals that are re-enqueued.

---

### 2.2 Live Intervals

**File:** `LiveIntervals.cpp`

Computes live ranges for virtual and physical registers.

| Function | Line | Purpose |
|----------|------|---------|
| `analyze()` | 159 | Main analysis entry point |
| `computeVirtRegInterval()` | 228 | Computes live interval for a virtual register |
| `computeVirtRegs()` | 236 | Computes all virtual register intervals |
| `computeRegMasks()` | 250 | Identifies call instructions that clobber registers |
| `computeRegUnitRange()` | 311 | Computes live range for a physical register unit |
| `computeLiveInRegUnits()` | 355 | Computes live-in register units from block live-ins |
| `shrinkToUses()` | 485 | **Shrinks a live interval** to actual uses. Essential after register coalescing. |
| `computeDeadValues()` | 543 | Identifies dead value numbers in a live interval |
| `extendToIndices()` | 656 | Extends a live range to cover specified slot indices |
| `pruneValue()` | 665 | Removes a live range segment after a kill point |
| `addKillFlags()` | 730 | Adds kill flags to last uses based on live intervals |
| `checkRegMaskInterference()` | 958 | Checks if live interval crosses a register clobber |
| `handleMove()` | 1562 | Updates intervals after instruction reordering |
| `splitSeparateComponents()` | 1805 | Splits interval with disconnected components |
| `constructMainRangeFromSubranges()` | 1821 | Rebuilds main range from subrange information |

---

### 2.3 PHI Elimination

**File:** `PHIElimination.cpp`

Replaces SSA phi nodes with copies on predecessor edges.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 230 | Entry point |
| `EliminatePHINodes()` | 313 | Per-block PHI elimination |
| `LowerPHINode()` | 361 | **Core algorithm**: replaces a single PHI with copies in each predecessor block. Handles identity copies, critical edge splitting, and live variable updates. |
| `SplitPHIEdges()` | 780 | Splits critical edges that would require copies on shared edges |
| `analyzePHINodes()` | 764 | Pre-analysis of PHI structure |
| `isLiveIn()` | 872 | Checks if register is live-in to a block |
| `isLiveOutPastPHIs()` | 881 | Checks if register is live-out past PHI definitions |

---

### 2.4 Prologue/Epilogue Inserter

**File:** `PrologEpilogInserter.cpp`

The target-independent framework that calls into target-specific frame lowering.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 214 | **Main pipeline**: calculateCallFrameInfo -> calculateSaveRestoreBlocks -> spillCalleeSavedRegs -> calculateFrameObjectOffsets -> insertPrologEpilogCode -> replaceFrameIndices |
| `calculateCallFrameInfo()` | 376 | Computes max call frame size from ADJCALLSTACKDOWN |
| `calculateSaveRestoreBlocks()` | 417 | Identifies blocks where callee-saves must be spilled/restored (entry + return blocks, plus shrink-wrapping) |
| `spillCalleeSavedRegs()` | 657 | Calls target to emit callee-saved spills and restores |
| `calculateFrameObjectOffsets()` | 859 | **Computes stack layout** — assigns concrete offsets to all stack objects (locals, spills, call args). Handles alignment, stack growth direction. |
| `insertPrologEpilogCode()` | 1170 | Calls target `emitPrologue()` and `emitEpilogue()` |
| `insertZeroCallUsedRegs()` | 1207 | Security: zero caller-saved regs before return |
| `replaceFrameIndices()` | 1379 | Replaces abstract frame indices with SP/FP+offset |
| `replaceFrameIndicesBackward()` | 1351 | Backward frame index replacement (for targets that prefer it) |
| `replaceFrameIndexDebugInstr()` | 1395 | Handles debug info frame index references |

---

### 2.5 Branch Relaxation

**File:** `BranchRelaxation.cpp`

Handles branches that exceed their encoding range by inserting trampolines.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 779 | Entry point |
| `scanFunction()` | 187 | Computes block offsets and sizes |
| `adjustBlockOffsets()` | 244 | Updates offsets after modification |
| `isBlockInRange()` | 351 | Checks if target block is within branch range |
| `fixupConditionalBranch()` | 376 | **Relaxes conditional branch**: inverts condition, inserts unconditional branch to far target |
| `fixupUnconditionalBranch()` | 576 | Relaxes unconditional branch (inserts trampoline or indirect branch) |
| `relaxBranchInstructions()` | 697 | Iterates until all branches are in range (fixpoint) |
| `verify()` | 148 | Verifies all branches are within range |

---

### 2.6 Machine Combiner

**File:** `MachineCombiner.cpp`

Post-isel peephole optimization that reassociates instructions to reduce critical path.

| Function | Line | Purpose |
|----------|------|---------|
| `runOnMachineFunction()` | 698 | Entry point |
| `combineInstructions()` | 522 | **Per-block combine loop**: queries target for combine patterns, evaluates profitability, applies. |
| `improvesCriticalPathLen()` | 347 | Checks if a combine reduces the critical path length |
| `preservesResourceLen()` | 421 | Checks if combine doesn't increase resource pressure |
| `reduceRegisterPressure()` | 331 | Checks if combine reduces register pressure |
| `isTransientMI()` | 155 | Identifies zero-latency instructions (copies, etc.) |

---

### 2.7 Machine CSE

**File:** `MachineCSE.cpp`

Machine-level common subexpression elimination.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 937 | Entry point |
| `ProcessBlockCSE()` | 526 | **Per-block CSE**: walks instructions, checks hash table for duplicates, replaces if profitable |
| `PerformCSE()` | 766 | Dominator-tree walk for CSE |
| `PerformSimplePRE()` | 901 | Partial redundancy elimination |
| `ProcessBlockPRE()` | 821 | Per-block PRE |
| `isCSECandidate()` | 398 | Determines if instruction can participate in CSE |
| `isProfitableToCSE()` | 435 | Checks if CSE replacement reduces register pressure |
| `PerformTrivialCopyPropagation()` | 173 | Propagates trivial copies |
| `hasLivePhysRegDefUses()` | 282 | Checks physical register interference for CSE safety |
| `PhysRegDefsReach()` | 332 | Checks if physical register definitions reach from CSE candidate to use |
| `isPRECandidate()` | 799 | Checks if instruction is a PRE candidate |
| `isProfitableToHoistInto()` | 919 | Checks if PRE hoisting is profitable |

---

### 2.8 Machine LICM

**File:** `MachineLICM.cpp`

Machine-level loop-invariant code motion.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 361 | Entry point |
| `HoistOutOfLoop()` | 807 | **Main hoisting walk** — dominator-tree traversal of loop body |
| `HoistRegionPostRA()` | 587 | Post-regalloc hoisting (more conservative) |
| `ProcessMI()` | 501 | Processes a single instruction for hoistability |
| `IsLICMCandidate()` | 1078 | Determines if instruction can be hoisted |
| `IsLoopInvariantInst()` | 1113 | Checks if all operands are loop-invariant |
| `IsProfitableToHoist()` | 1253 | **Profitability heuristic**: considers register pressure, execution frequency, CSE opportunities |
| `IsGuaranteedToExecute()` | 751 | Checks if block executes on every loop iteration |
| `HasLoopPHIUse()` | 1124 | Checks for loop-carried PHI uses |
| `HasHighOperandLatency()` | 1158 | Checks operand latency for hoisting benefit |
| `IsCheapInstruction()` | 1190 | Identifies trivially cheap instructions |
| `CanCauseHighRegPressure()` | 1215 | Estimates register pressure impact |
| `EliminateCSE()` | 1505 | Eliminates CSE opportunities after hoisting |
| `MayCSE()` | 1570 | Checks if hoisted instruction matches existing |
| `InitCSEMap()` | 1445 | Initializes CSE lookup for loop preheader |

---

### 2.9 Machine Block Placement

**File:** `MachineBlockPlacement.cpp`

Basic block layout optimization for I-cache and branch prediction.

| Function | Line | Purpose |
|----------|------|---------|
| `run()` | 3564 | Entry point |
| `buildCFGChains()` | 2792 | **Main algorithm**: builds chains of blocks connected by fall-through edges |
| `buildChain()` | 1913 | Extends a single chain by selecting best successors |
| `buildLoopChains()` | 2699 | Arranges blocks within a loop |
| `hasBetterLayoutPredecessor()` | 1473 | Checks if block should break chain for better predecessor |
| `rotateLoop()` | 2416 | Rotates loop to put best exit at bottom |
| `rotateLoopWithProfile()` | 2504 | Profile-guided loop rotation |
| `optimizeBranches()` | 2957 | Eliminates unnecessary branches after layout |
| `alignBlocks()` | 2996 | Inserts alignment for hot blocks and loop headers |
| `isTrellis()` | 1021 | Detects trellis patterns in CFG |
| `shouldTailDuplicate()` | 838 | Decides whether to tail-duplicate a block |
| `maybeTailDuplicateBlock()` | 3209 | Attempts tail duplication for layout improvement |
| `applyExtTsp()` | 3676 | Applies ExtTSP (extended traveling salesman) algorithm |

---

## 3. MC Layer (`llvm/lib/MC/`)

### 3.1 Mach-O Object Writer

**File:** `MachObjectWriter.cpp`

Writes the complete Mach-O binary format.

| Function | Line | Purpose |
|----------|------|---------|
| `writeObject()` | 795 | **Main entry point** — orchestrates the entire Mach-O file output. Computes symbol table, section layout, load commands, then writes header, segments, sections, relocations, symbols, string table. |
| `writeHeader()` | 170 | Writes Mach-O header (magic, CPU type, file type, load command count) |
| `writeSegmentLoadCommand()` | 222 | Writes LC_SEGMENT_64 load command |
| `writeSection()` | 262 | Writes section header (name, segment, addr, offset, align, relocs) |
| `writeNlist()` | 385 | Writes a single symbol table entry (nlist_64) |
| `writeSymtabLoadCommand()` | 306 | Writes LC_SYMTAB load command |
| `writeDysymtabLoadCommand()` | 325 | Writes LC_DYSYMTAB load command |
| `writeLinkeditLoadCommand()` | 453 | Writes linkedit data load command (code signature, etc.) |
| `computeSymbolTable()` | 578 | **Builds symbol table** — sorts symbols into local/external/undefined, assigns indices, builds string table |
| `computeSectionAddresses()` | 687 | Assigns virtual addresses to sections |
| `recordRelocation()` | 508 | Dispatches to target-specific relocation recording |
| `bindIndirectSymbols()` | 521 | Binds indirect symbols for lazy/non-lazy symbol pointers |
| `getSymbolAddress()` | 94 | Computes final address for a symbol |
| `executePostLayoutBinding()` | 717 | Post-layout fixups |
| `isSymbolRefDifferenceFullyResolvedImpl()` | 724 | Checks if symbol difference can be resolved at assembly time |
| `reset()` | 43 | Resets writer state for new object |

---

### 3.2 MCExpr (Assembly Expressions)

**File:** `llvm/include/llvm/MC/MCExpr.h` (line 34)

Base class for assembly-level expressions used in fixups and relocations.

| ExprKind | Line | Purpose |
|----------|------|---------|
| `Binary` | 41 | Binary expressions (add, sub, etc.) |
| `Constant` | 42 | Integer constants |
| `SymbolRef` | 43 | Symbol references (labels, external symbols) |
| `Unary` | 44 | Unary expressions (negate, complement) |
| `Specifier` | 45 | Expressions with relocation specifiers (e.g., `:got:`, `:lo12:`) |
| `Target` | 46 | Target-specific expression extensions |

---

### 3.3 MCFixup (Relocations)

**File:** `llvm/include/llvm/MC/MCFixup.h` (line 61)

Represents a fixup that may become a relocation in the object file.

**Key fields:**
- `Value` (MCExpr*) — The expression to evaluate for the fixup
- `Offset` (uint32_t) — Byte offset within the fragment
- `Kind` (MCFixupKind) — Type of fixup (determines encoding)
- `PCRel` (bool) — Whether this is PC-relative
- `LinkerRelaxable` (bool) — Whether linker can relax this

**Standard fixup kinds** (line 23):
| Kind | Purpose |
|------|---------|
| `FK_Data_1` | 1-byte data fixup |
| `FK_Data_2` | 2-byte data fixup |
| `FK_Data_4` | 4-byte data fixup |
| `FK_Data_8` | 8-byte data fixup |
| `FK_NONE` | No-op fixup |
| `FirstTargetFixupKind` | Start of target-specific fixup kinds |

---

## 4. Machine IR Model (`llvm/include/llvm/CodeGen/`)

### 4.1 MachineInstr

**File:** `MachineInstr.h` (line 71)

Represents a single machine instruction.

**Key fields:**
- `MCID` (MCInstrDesc*) — Instruction descriptor (opcode, flags, operand info)
- `Parent` (MachineBasicBlock*) — Owning basic block
- `Operands` (MachineOperand*) — Array of operands

**MIFlags** (line 88): Bitfield for instruction properties:
- `FrameSetup` / `FrameDestroy` — Prologue/epilogue markers
- `BundledPred` / `BundledSucc` — Bundle membership
- `FmNoNans` / `FmNoInfs` / `FmNsz` / `FmArcp` / `FmContract` / `FmAfn` / `FmReassoc` — Fast-math flags
- `NoUWrap` / `NoSWrap` / `IsExact` — Integer overflow flags
- `NoFPExcept` — FP exception control
- `NoMerge` — Prevent branch folding
- `LRSplit` — Live range split marker

### 4.2 MachineOperand

**File:** `MachineOperand.h` (line 49)

Represents a single instruction operand.

**MachineOperandType** enum (line 51):
| Type | Purpose |
|------|---------|
| `MO_Register` | Virtual or physical register |
| `MO_Immediate` | Integer immediate (up to 64 bits) |
| `MO_CImmediate` | Immediate >64 bits |
| `MO_FPImmediate` | Floating-point immediate |
| `MO_MachineBasicBlock` | Branch target block |
| `MO_FrameIndex` | Abstract stack frame index |
| `MO_ConstantPoolIndex` | Constant pool entry |
| `MO_TargetIndex` | Target-specific index |
| `MO_JumpTableIndex` | Jump table entry |
| `MO_ExternalSymbol` | External symbol name |
| `MO_GlobalAddress` | Global value address |
| `MO_BlockAddress` | Basic block address |
| `MO_RegisterMask` | Call clobber mask |
| `MO_RegisterLiveOut` | Live-out register mask |
| `MO_Metadata` | Debug metadata |
| `MO_MCSymbol` | MC-level symbol |
| `MO_CFIIndex` | CFI instruction index |
| `MO_IntrinsicID` | Intrinsic ID for ISel |
| `MO_Predicate` | Generic predicate |
| `MO_ShuffleMask` | Shuffle mask constant |

**Register operand flags** (bit fields, lines 93-115):
- `IsDef` / `IsImp` — Definition / implicit
- `IsDeadOrKill` — Dead def or last use
- `IsRenamable` — Can be renamed by regalloc
- `IsUndef` — Reads undef value
- `SubReg_TargetFlags` — Sub-register index (12 bits)
- `TiedTo` — Tied operand index (4 bits)

### 4.3 MachineBasicBlock

**File:** `MachineBasicBlock.h` (line 121)

Represents a basic block in machine code.

**Key fields:**
- `BB` (BasicBlock*) — Corresponding IR basic block (may be null)
- `Number` (int) — Block number in function
- `Insts` (ilist<MachineInstr>) — Instruction list
- `Predecessors` / `Successors` — CFG edges (SmallVector)
- `Probs` — Branch probabilities for successors
- `LiveIns` — Physical registers live at block entry
- `Alignment` — Required alignment
- `IsEHPad` — Exception handler entry
- `CallFrameSize` — Outstanding call frame setup size

### 4.4 MachineFunction

**File:** `MachineFunction.h` (line 61+)

Top-level container for a function's machine code.

**Key types:**
- `MachineFunctionInfo` (line 105) — Target-specific per-function info (e.g., `AArch64FunctionInfo`)
- `MachineFunctionProperties` (line 138) — Tracks pipeline state: `IsSSA`, `NoPHIs`, `TracksLiveness`, `AllVRegsAllocated`, etc.

---

## 5. Porting Priority and Insights

### Critical Path (Minimum Viable Backend)

1. **Immediate materialization** (`AArch64ExpandImm.cpp`) — Small, self-contained, well-documented. Port first. The logical immediate encoding (`processLogicalImmediate`) is the tricky part.

2. **Register file definition** (`AArch64RegisterInfo.td`) — Define GPR32/GPR64/FPR* register classes, sub-register relationships, and calling convention registers.

3. **Calling convention** (`AArch64CallingConvention.td`) — Start with DarwinPCS (our primary target). GPR: X0-X7, FPR: D0-D7, return in X0/D0.

4. **Binary encoding** (`AArch64MCCodeEmitter.cpp`) — 784 lines. Most encoding is mechanical (TableGen-generated in LLVM). For LLVM2, we generate encoding directly from instruction definitions.

5. **Frame lowering** (`AArch64FrameLowering.cpp` + `AArch64PrologueEpilogue.cpp`) — Prologue/epilogue emission, callee-save spilling. The compact unwind encoding is well-defined and small.

6. **Fixup/Relocation model** (`AArch64FixupKinds.h` + `AArch64MachObjectWriter.cpp`) — Only 12 fixup kinds total. The Mach-O relocation mapping is compact.

7. **Mach-O writer** (`MachObjectWriter.cpp`) — Structured and well-documented. The `writeObject()` function at line 795 is the complete orchestration.

### Key Algorithmic Insights

- **Register allocation** (RegAllocGreedy): The cascade is `tryAssign -> tryEvict -> trySplit -> tryLastChanceRecoloring`. For LLVM2 MVP, a simpler linear-scan allocator may suffice initially, with greedy as an optimization target.

- **Branch relaxation** is a fixpoint algorithm: scan, check ranges, insert trampolines, repeat. AArch64 branch ranges: B.cond = +/-1MB (19-bit), B = +/-128MB (26-bit), TBZ = +/-32KB (14-bit).

- **Load/store optimization** (LDP/STP merging) is one of the highest-impact AArch64-specific optimizations. The algorithm is: scan for adjacent load/store pairs with consecutive addresses, merge into paired instructions.

- **Block placement** uses chain-based layout: build chains of fall-through blocks, then arrange chains to minimize taken branches. The ExtTSP algorithm (line 3676) is the state of the art.

- **DAG combines** (`PerformDAGCombine` at line 29523) is where most target-specific peephole optimizations live. This is a massive function but individual combines are self-contained.

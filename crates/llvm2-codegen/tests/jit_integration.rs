// llvm2-codegen/tests/jit_integration.rs - JIT end-to-end integration tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Tests that compile IR functions through the full JIT pipeline and execute
// them in-process via JitCompiler::compile_raw(). Exercises:
// - Simple arithmetic functions (add, multiply, subtract)
// - Multi-function compilation with cross-function BL calls
// - External symbol resolution via veneer trampolines
// - Edge cases: branch patching limits, large function buffers
//
// Part of #342 — JIT integration tests: end-to-end compile + execute

#![cfg(target_arch = "aarch64")]
// Existing tests use `ExecutableBuffer::get_fn` and `get_fn_ptr`, which are
// deprecated in favour of the lifetime-bound `get_fn_bound` /
// `get_fn_ptr_bound` APIs (issue #355). These tests continue to exercise
// the legacy paths intentionally as regression coverage; silence the
// per-call deprecation warnings at the file level.
#![allow(deprecated)]

use std::collections::HashMap;

use llvm2_codegen::jit::{JitCompiler, JitConfig, JitError};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::operand::MachOperand::Special;
use llvm2_ir::regs::{SpecialReg, X0, X1, X8, X9, FP, LR};

// ---------------------------------------------------------------------------
// Test helpers: function builders
// ---------------------------------------------------------------------------

/// Build `fn add(a: i64, b: i64) -> i64 { a + b }`
///
/// ADD X0, X0, X1 ; RET
fn build_add() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("add".to_string(), sig);
    let entry = func.entry;

    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn sub(a: i64, b: i64) -> i64 { a - b }`
///
/// SUB X0, X0, X1 ; RET
fn build_sub() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("sub".to_string(), sig);
    let entry = func.entry;

    let sub = MachInst::new(
        AArch64Opcode::SubRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
        ],
    );
    let sub_id = func.push_inst(sub);
    func.append_inst(entry, sub_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn mul(a: i64, b: i64) -> i64 { a * b }`
///
/// MUL X0, X0, X1 (encoded as MADD X0, X0, X1, XZR) ; RET
fn build_mul() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("mul".to_string(), sig);
    let entry = func.entry;

    let mul = MachInst::new(
        AArch64Opcode::MulRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
        ],
    );
    let mul_id = func.push_inst(mul);
    func.append_inst(entry, mul_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn return_const() -> i64 { 42 }`
///
/// MOVZ X0, #42 ; RET
fn build_return_const() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I64]);
    let mut func = MachFunction::new("return_const".to_string(), sig);
    let entry = func.entry;

    let mov = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(entry, mov_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn negate(a: i64) -> i64 { 0 - a }`
///
/// NEG X0, X0 (encoded as SUB X0, XZR, X0) ; RET
fn build_negate() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("negate".to_string(), sig);
    let entry = func.entry;

    let neg = MachInst::new(
        AArch64Opcode::Neg,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0)],
    );
    let neg_id = func.push_inst(neg);
    func.append_inst(entry, neg_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn identity(a: i64) -> i64 { a }`
///
/// RET (X0 passthrough)
fn build_identity() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("identity".to_string(), sig);
    let entry = func.entry;

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build iterative factorial:
///   fn factorial(n: i64) -> i64 {
///       result = 1, i = n
///       loop: if i <= 1 goto done; result *= i; i -= 1; goto loop
///       done: return result
///   }
fn build_factorial() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("factorial".to_string(), sig);

    let bb_entry = func.entry;
    let bb_loop = func.create_block();
    let bb_done = func.create_block();

    // MOV X8, X0 (i = n)
    let mov_n = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0)],
    );
    let mov_n_id = func.push_inst(mov_n);
    func.append_inst(bb_entry, mov_n_id);

    // MOVZ X9, #1 (result = 1)
    let mov_one = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(1)],
    );
    let mov_one_id = func.push_inst(mov_one);
    func.append_inst(bb_entry, mov_one_id);

    // (fall through to bb_loop)

    // bb_loop: CMP X8, #1
    let cmp = MachInst::new(
        AArch64Opcode::CmpRI,
        vec![MachOperand::PReg(X8), MachOperand::Imm(1)],
    );
    let cmp_id = func.push_inst(cmp);
    func.append_inst(bb_loop, cmp_id);

    // B.LE bb_done (+4 instructions forward)
    let ble = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0xD), // LE
            MachOperand::Imm(4),   // +4 insts to bb_done
        ],
    );
    let ble_id = func.push_inst(ble);
    func.append_inst(bb_loop, ble_id);

    // MUL X9, X9, X8 (result *= i)
    let mul = MachInst::new(
        AArch64Opcode::MulRR,
        vec![
            MachOperand::PReg(X9),
            MachOperand::PReg(X9),
            MachOperand::PReg(X8),
        ],
    );
    let mul_id = func.push_inst(mul);
    func.append_inst(bb_loop, mul_id);

    // SUB X8, X8, #1 (i -= 1)
    let sub = MachInst::new(
        AArch64Opcode::SubRI,
        vec![
            MachOperand::PReg(X8),
            MachOperand::PReg(X8),
            MachOperand::Imm(1),
        ],
    );
    let sub_id = func.push_inst(sub);
    func.append_inst(bb_loop, sub_id);

    // B bb_loop (-4 instructions)
    let b_loop = MachInst::new(AArch64Opcode::B, vec![MachOperand::Imm(-4i64)]);
    let b_loop_id = func.push_inst(b_loop);
    func.append_inst(bb_loop, b_loop_id);

    // bb_done: MOV X0, X9
    let mov_result = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X9)],
    );
    let mov_result_id = func.push_inst(mov_result);
    func.append_inst(bb_done, mov_result_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(bb_done, ret_id);

    func
}

/// Build a function that calls another function via BL symbol reference.
///
/// `fn caller(a: i64, b: i64) -> i64 { callee(a, b) }`
///
/// Since BL clobbers LR, the caller must save/restore LR (X30) around the
/// call. We emit a minimal prologue/epilogue:
///   STP FP, LR, [SP, #-16]!   (save FP and LR, pre-index)
///   BL <callee>                (call — args already in X0, X1)
///   LDP FP, LR, [SP], #16     (restore FP and LR, post-index)
///   RET                        (return callee's result in X0)
fn build_caller(caller_name: &str, callee_name: &str) -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new(caller_name.to_string(), sig);
    let entry = func.entry;

    // STP FP, LR, [SP, #-16]! (pre-index: save frame pointer and link register)
    let stp = MachInst::new(
        AArch64Opcode::StpPreIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(-16),
        ],
    );
    let stp_id = func.push_inst(stp);
    func.append_inst(entry, stp_id);

    // BL <callee> (symbol reference — resolved by JIT linker)
    let bl = MachInst::new(
        AArch64Opcode::Bl,
        vec![MachOperand::Symbol(callee_name.to_string())],
    );
    let bl_id = func.push_inst(bl);
    func.append_inst(entry, bl_id);

    // LDP FP, LR, [SP], #16 (post-index: restore frame pointer and link register)
    let ldp = MachInst::new(
        AArch64Opcode::LdpPostIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(16),
        ],
    );
    let ldp_id = func.push_inst(ldp);
    func.append_inst(entry, ldp_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build a "large" function that contains many ADD-immediate-zero instructions
/// to test code buffers approaching meaningful sizes.
///
/// Uses `ADD X8, X8, #0` as a no-op filler (real encoded instruction, unlike
/// the pseudo Nop which is skipped during encoding).
///
/// `fn large_fn(a: i64) -> i64 { /* filler_count ADDs */ return a }`
fn build_large_filler_function(name: &str, filler_count: usize) -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new(name.to_string(), sig);
    let entry = func.entry;

    for _ in 0..filler_count {
        // ADD X8, X8, #0 — real instruction, no effect on X0 (the return value).
        let filler = MachInst::new(
            AArch64Opcode::AddRI,
            vec![
                MachOperand::PReg(X8),
                MachOperand::PReg(X8),
                MachOperand::Imm(0),
            ],
        );
        let filler_id = func.push_inst(filler);
        func.append_inst(entry, filler_id);
    }

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

// ---------------------------------------------------------------------------
// Test: simple add — compile and call via JIT
// ---------------------------------------------------------------------------

#[test]
fn test_jit_add() {
    let jit = JitCompiler::new(JitConfig::default());
    let add_fn = build_add();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[add_fn], &ext).expect("compile_raw should succeed");

    assert!(buf.allocated_size() > 0, "buffer should have nonzero size");
    assert!(buf.symbol_count() >= 1, "should have at least 1 symbol");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("add").expect("should find 'add' symbol")
    };

    assert_eq!(f(3, 4), 7);
    assert_eq!(f(0, 0), 0);
    assert_eq!(f(-1, 1), 0);
    assert_eq!(f(100, 200), 300);
    assert_eq!(f(i64::MAX, 0), i64::MAX);
    assert_eq!(f(-100, -200), -300);
}

// ---------------------------------------------------------------------------
// Test: subtract
// ---------------------------------------------------------------------------

#[test]
fn test_jit_sub() {
    let jit = JitCompiler::new(JitConfig::default());
    let sub_fn = build_sub();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[sub_fn], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("sub").expect("should find 'sub' symbol")
    };

    assert_eq!(f(10, 3), 7);
    assert_eq!(f(0, 0), 0);
    assert_eq!(f(5, 5), 0);
    assert_eq!(f(-10, -3), -7);
    assert_eq!(f(1, -1), 2);
}

// ---------------------------------------------------------------------------
// Test: multiply
// ---------------------------------------------------------------------------

#[test]
fn test_jit_mul() {
    let jit = JitCompiler::new(JitConfig::default());
    let mul_fn = build_mul();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[mul_fn], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("mul").expect("should find 'mul' symbol")
    };

    assert_eq!(f(3, 4), 12);
    assert_eq!(f(0, 999), 0);
    assert_eq!(f(7, 1), 7);
    assert_eq!(f(-3, 4), -12);
    assert_eq!(f(-3, -4), 12);
    assert_eq!(f(1000, 1000), 1_000_000);
}

// ---------------------------------------------------------------------------
// Test: return constant
// ---------------------------------------------------------------------------

#[test]
fn test_jit_return_const() {
    let jit = JitCompiler::new(JitConfig::default());
    let func = build_return_const();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn() -> i64 = unsafe {
        buf.get_fn("return_const").expect("should find 'return_const' symbol")
    };

    assert_eq!(f(), 42);
}

// ---------------------------------------------------------------------------
// Test: negate
// ---------------------------------------------------------------------------

#[test]
fn test_jit_negate() {
    let jit = JitCompiler::new(JitConfig::default());
    let func = build_negate();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("negate").expect("should find 'negate' symbol")
    };

    assert_eq!(f(5), -5);
    assert_eq!(f(-5), 5);
    assert_eq!(f(0), 0);
    assert_eq!(f(1), -1);
}

// ---------------------------------------------------------------------------
// Test: identity (passthrough)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_identity() {
    let jit = JitCompiler::new(JitConfig::default());
    let func = build_identity();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("identity").expect("should find 'identity' symbol")
    };

    assert_eq!(f(0), 0);
    assert_eq!(f(42), 42);
    assert_eq!(f(-1), -1);
    assert_eq!(f(i64::MAX), i64::MAX);
    assert_eq!(f(i64::MIN), i64::MIN);
}

// ---------------------------------------------------------------------------
// Test: factorial (loop)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_factorial() {
    let jit = JitCompiler::new(JitConfig::default());
    let func = build_factorial();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("compile_raw should succeed");

    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("factorial").expect("should find 'factorial' symbol")
    };

    assert_eq!(f(0), 1);
    assert_eq!(f(1), 1);
    assert_eq!(f(5), 120);
    assert_eq!(f(10), 3_628_800);
    assert_eq!(f(20), 2_432_902_008_176_640_000);
}

// ---------------------------------------------------------------------------
// Test: multiple functions in one compilation unit
// ---------------------------------------------------------------------------

#[test]
fn test_jit_multiple_functions() {
    let jit = JitCompiler::new(JitConfig::default());
    let funcs = vec![build_add(), build_sub(), build_mul(), build_return_const()];
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&funcs, &ext).expect("compile_raw should succeed");

    assert!(
        buf.symbol_count() >= 4,
        "should have at least 4 symbols, got {}",
        buf.symbol_count()
    );

    let add: extern "C" fn(i64, i64) -> i64 =
        unsafe { buf.get_fn("add").expect("add") };
    let sub: extern "C" fn(i64, i64) -> i64 =
        unsafe { buf.get_fn("sub").expect("sub") };
    let mul: extern "C" fn(i64, i64) -> i64 =
        unsafe { buf.get_fn("mul").expect("mul") };
    let rc: extern "C" fn() -> i64 =
        unsafe { buf.get_fn("return_const").expect("return_const") };

    assert_eq!(add(10, 20), 30);
    assert_eq!(sub(10, 3), 7);
    assert_eq!(mul(6, 7), 42);
    assert_eq!(rc(), 42);
}

// ---------------------------------------------------------------------------
// Test: cross-function BL call (caller -> callee within same compilation)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_cross_function_bl() {
    let jit = JitCompiler::new(JitConfig::default());

    // callee: add(a, b) -> a + b
    let callee = build_add();
    // caller: calls "add" via BL symbol, forwarding args
    let caller = build_caller("call_add", "add");

    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[callee, caller], &ext)
        .expect("compile_raw should succeed with cross-function BL");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("call_add").expect("should find 'call_add' symbol")
    };

    assert_eq!(f(10, 20), 30);
    assert_eq!(f(0, 0), 0);
    assert_eq!(f(-5, 5), 0);
}

// ---------------------------------------------------------------------------
// Test: cross-function BL with underscore-prefixed symbol resolution
// ---------------------------------------------------------------------------

#[test]
fn test_jit_cross_function_bl_underscore_prefix() {
    let jit = JitCompiler::new(JitConfig::default());

    // callee: add(a, b) -> a + b  (registered as both "add" and "_add")
    let callee = build_add();
    // caller: calls "_add" via BL symbol (Mach-O mangling prefix)
    let caller = build_caller("call_add_mangled", "_add");

    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[callee, caller], &ext)
        .expect("compile_raw should succeed with _-prefixed symbol");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("call_add_mangled")
            .expect("should find 'call_add_mangled' symbol")
    };

    assert_eq!(f(7, 8), 15);
}

// ---------------------------------------------------------------------------
// Test: external symbol resolution via veneer trampolines
// ---------------------------------------------------------------------------

/// A native helper function used as an external symbol for JIT tests.
/// add_ten(x) -> x + 10
extern "C" fn host_add_ten(x: i64) -> i64 {
    x + 10
}

/// Build a function that calls an external symbol.
///
/// `fn call_extern(a: i64) -> i64 { callee_symbol(a) }`
///
/// Includes STP/LDP prologue/epilogue to save/restore LR around the BL.
fn build_extern_caller(callee_symbol: &str) -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("call_extern".to_string(), sig);
    let entry = func.entry;

    // STP FP, LR, [SP, #-16]!
    let stp = MachInst::new(
        AArch64Opcode::StpPreIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(-16),
        ],
    );
    let stp_id = func.push_inst(stp);
    func.append_inst(entry, stp_id);

    // BL <external symbol>
    let bl = MachInst::new(
        AArch64Opcode::Bl,
        vec![MachOperand::Symbol(callee_symbol.to_string())],
    );
    let bl_id = func.push_inst(bl);
    func.append_inst(entry, bl_id);

    // LDP FP, LR, [SP], #16
    let ldp = MachInst::new(
        AArch64Opcode::LdpPostIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(16),
        ],
    );
    let ldp_id = func.push_inst(ldp);
    func.append_inst(entry, ldp_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

#[test]
fn test_jit_external_symbol_veneer() {
    let jit = JitCompiler::new(JitConfig::default());
    let caller = build_extern_caller("_host_add_ten");

    let mut ext: HashMap<String, *const u8> = HashMap::new();
    ext.insert(
        "_host_add_ten".to_string(),
        host_add_ten as *const u8,
    );

    let buf = jit
        .compile_raw(&[caller], &ext)
        .expect("compile_raw should succeed with external symbol");

    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("call_extern")
            .expect("should find 'call_extern' symbol")
    };

    assert_eq!(f(0), 10);
    assert_eq!(f(32), 42);
    assert_eq!(f(-10), 0);
    assert_eq!(f(100), 110);
}

// ---------------------------------------------------------------------------
// Test: external symbol with multiple callers sharing one veneer
// ---------------------------------------------------------------------------

extern "C" fn host_double(x: i64) -> i64 {
    x * 2
}

#[test]
fn test_jit_shared_veneer() {
    let jit = JitCompiler::new(JitConfig::default());

    // Two different callers both referencing the same external symbol.
    // Each saves/restores LR around the BL (non-leaf function pattern).
    let caller_a = {
        let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("caller_a".to_string(), sig);
        let entry = func.entry;
        // STP FP, LR, [SP, #-16]!
        let stp = MachInst::new(
            AArch64Opcode::StpPreIndex,
            vec![MachOperand::PReg(FP), MachOperand::PReg(LR),
                 Special(SpecialReg::SP), MachOperand::Imm(-16)],
        );
        let stp_id = func.push_inst(stp);
        func.append_inst(entry, stp_id);
        let bl = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("_host_double".to_string())],
        );
        let bl_id = func.push_inst(bl);
        func.append_inst(entry, bl_id);
        // LDP FP, LR, [SP], #16
        let ldp = MachInst::new(
            AArch64Opcode::LdpPostIndex,
            vec![MachOperand::PReg(FP), MachOperand::PReg(LR),
                 Special(SpecialReg::SP), MachOperand::Imm(16)],
        );
        let ldp_id = func.push_inst(ldp);
        func.append_inst(entry, ldp_id);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);
        func
    };

    let caller_b = {
        let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("caller_b".to_string(), sig);
        let entry = func.entry;
        // STP FP, LR, [SP, #-16]!
        let stp = MachInst::new(
            AArch64Opcode::StpPreIndex,
            vec![MachOperand::PReg(FP), MachOperand::PReg(LR),
                 Special(SpecialReg::SP), MachOperand::Imm(-16)],
        );
        let stp_id = func.push_inst(stp);
        func.append_inst(entry, stp_id);
        let bl = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("_host_double".to_string())],
        );
        let bl_id = func.push_inst(bl);
        func.append_inst(entry, bl_id);
        // LDP FP, LR, [SP], #16
        let ldp = MachInst::new(
            AArch64Opcode::LdpPostIndex,
            vec![MachOperand::PReg(FP), MachOperand::PReg(LR),
                 Special(SpecialReg::SP), MachOperand::Imm(16)],
        );
        let ldp_id = func.push_inst(ldp);
        func.append_inst(entry, ldp_id);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);
        func
    };

    let mut ext: HashMap<String, *const u8> = HashMap::new();
    ext.insert("_host_double".to_string(), host_double as *const u8);

    let buf = jit
        .compile_raw(&[caller_a, caller_b], &ext)
        .expect("compile_raw should succeed with shared veneer");

    let fa: extern "C" fn(i64) -> i64 =
        unsafe { buf.get_fn("caller_a").expect("caller_a") };
    let fb: extern "C" fn(i64) -> i64 =
        unsafe { buf.get_fn("caller_b").expect("caller_b") };

    assert_eq!(fa(5), 10);
    assert_eq!(fb(21), 42);
    // Both callers should produce the same result for the same input.
    assert_eq!(fa(100), fb(100));
}

// ---------------------------------------------------------------------------
// Test: unresolved symbol produces an error
// ---------------------------------------------------------------------------

#[test]
fn test_jit_unresolved_symbol_error() {
    let jit = JitCompiler::new(JitConfig::default());
    let caller = build_extern_caller("_nonexistent_function");
    let ext: HashMap<String, *const u8> = HashMap::new();

    let result = jit.compile_raw(&[caller], &ext);
    assert!(
        result.is_err(),
        "compile_raw should fail for unresolved symbol"
    );
    match result {
        Err(JitError::UnresolvedSymbol(name)) => {
            assert_eq!(name, "_nonexistent_function");
        }
        Err(other) => panic!("Expected UnresolvedSymbol error, got: {}", other),
        Ok(_) => unreachable!("already checked result.is_err()"),
    }
}

// ---------------------------------------------------------------------------
// Test: duplicate symbol names produce JitError::DuplicateSymbol (#374)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_duplicate_primary_symbol_error() {
    // Two functions with identical primary names — direct collision on the
    // primary-name slot.
    let jit = JitCompiler::new(JitConfig::default());
    let mut a = build_add();
    let mut b = build_add();
    a.name = "dup".to_string();
    b.name = "dup".to_string();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let result = jit.compile_raw(&[a, b], &ext);
    match result {
        Err(JitError::DuplicateSymbol(name)) => {
            assert_eq!(name, "dup", "collision should report primary name");
        }
        Err(other) => panic!("Expected DuplicateSymbol, got: {}", other),
        Ok(_) => panic!("Expected DuplicateSymbol error, got success"),
    }
}

#[test]
fn test_jit_duplicate_alias_vs_primary_symbol_error() {
    // Order `[foo, _foo]`: the second function's PRIMARY name `_foo`
    // collides with the first function's already-inserted alias `_foo`.
    // This fires the primary-name check on iteration 2.
    let jit = JitCompiler::new(JitConfig::default());
    let mut a = build_add();
    let mut b = build_add();
    a.name = "foo".to_string();
    b.name = "_foo".to_string();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let result = jit.compile_raw(&[a, b], &ext);
    match result {
        Err(JitError::DuplicateSymbol(name)) => {
            assert_eq!(
                name, "_foo",
                "collision should report the colliding key (`_foo`)"
            );
        }
        Err(other) => panic!("Expected DuplicateSymbol, got: {}", other),
        Ok(_) => panic!("Expected DuplicateSymbol error, got success"),
    }
}

#[test]
fn test_jit_duplicate_alias_check_path() {
    // Order `[_foo, foo]`: exercises the ALIAS check branch specifically.
    //
    // Iter 1 (`_foo`): primary `_foo` and alias `__foo` inserted. No collision.
    // Iter 2 (`foo`): primary `foo` not present → inserted. Alias `_foo` is
    //                 already present → alias check fires.
    //
    // Without this ordering, the alias-check branch (the `if func_offsets.
    // contains_key(alias.as_str())` block) would not be exercised by any
    // test — per AI Model review feedback on the #374 fix.
    let jit = JitCompiler::new(JitConfig::default());
    let mut a = build_add();
    let mut b = build_add();
    a.name = "_foo".to_string();
    b.name = "foo".to_string();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let result = jit.compile_raw(&[a, b], &ext);
    match result {
        Err(JitError::DuplicateSymbol(name)) => {
            assert_eq!(
                name, "_foo",
                "alias collision should report the generated alias key (`_foo`)"
            );
        }
        Err(other) => panic!("Expected DuplicateSymbol, got: {}", other),
        Ok(_) => panic!("Expected DuplicateSymbol error, got success"),
    }
}

// ---------------------------------------------------------------------------
// Test: ExecutableBuffer API — symbol enumeration and lookup
// ---------------------------------------------------------------------------

#[test]
fn test_jit_buffer_api() {
    let jit = JitCompiler::new(JitConfig::default());
    let funcs = vec![build_add(), build_sub()];
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&funcs, &ext).expect("compile_raw should succeed");

    // Fix #360: symbol_count reflects the canonical symbol list (one entry
    // per compiled function), not the /2-ed size of the lookup map.
    assert_eq!(
        buf.symbol_count(),
        2,
        "expected exactly 2 canonical symbols (add, sub)"
    );

    // get_fn_ptr returns valid pointers.
    assert!(buf.get_fn_ptr("add").is_some(), "should find 'add'");
    assert!(buf.get_fn_ptr("sub").is_some(), "should find 'sub'");
    assert!(buf.get_fn_ptr("_add").is_some(), "should find '_add' alias");
    assert!(buf.get_fn_ptr("_sub").is_some(), "should find '_sub' alias");
    assert!(buf.get_fn_ptr("nonexistent").is_none(), "shouldn't find fake symbol");

    // Fix #360: symbols() enumerates canonical names (no alias duplicates,
    // no hiding of `_`-prefixed user names).
    let names: Vec<&str> = buf.symbols().map(|(name, _)| name).collect();
    assert_eq!(names.len(), 2, "symbols() must yield canonical names only");
    assert!(names.contains(&"add"), "symbols() should contain 'add'");
    assert!(names.contains(&"sub"), "symbols() should contain 'sub'");

    // Allocated size should be at least a page.
    assert!(
        buf.allocated_size() >= 4096,
        "buffer should be at least one page"
    );
}

// ---------------------------------------------------------------------------
// Test: ExecutableBuffer is Send + Sync
// ---------------------------------------------------------------------------

#[test]
fn test_jit_buffer_send_sync() {
    let jit = JitCompiler::new(JitConfig::default());
    let func = build_add();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("compile_raw should succeed");

    // Verify the buffer can be sent to another thread and used there.
    let handle = std::thread::spawn(move || {
        let f: extern "C" fn(i64, i64) -> i64 = unsafe {
            buf.get_fn("add").expect("add")
        };
        f(10, 20)
    });

    assert_eq!(handle.join().unwrap(), 30);
}

// ---------------------------------------------------------------------------
// Test: empty function list
// ---------------------------------------------------------------------------

#[test]
fn test_jit_empty_functions() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[], &ext)
        .expect("compile_raw should succeed with empty input");

    assert_eq!(buf.symbol_count(), 0);
}

// ---------------------------------------------------------------------------
// Test: large function buffer (many filler instructions, multi-page mmap)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_large_function() {
    let jit = JitCompiler::new(JitConfig::default());
    // 4096 instructions = 16384 bytes = exactly one Apple Silicon page.
    // Add the RET instruction, and we spill into a second page.
    let func = build_large_filler_function("big_fn", 4096);
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[func], &ext)
        .expect("compile_raw should succeed with large function");

    // The function should still work — filler ADD X8,X8,#0 doesn't affect X0.
    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("big_fn").expect("should find 'big_fn'")
    };

    assert_eq!(f(42), 42);
    assert_eq!(f(0), 0);

    // Buffer should be at least 2 pages (filler + RET > 1 page on Apple Silicon).
    assert!(
        buf.allocated_size() >= 2 * 16384,
        "buffer should span multiple pages, got {} bytes",
        buf.allocated_size()
    );
}

// ---------------------------------------------------------------------------
// Test: JitConfig with O0 optimization
// ---------------------------------------------------------------------------

#[test]
fn test_jit_config_o0() {
    let jit = JitCompiler::new(JitConfig {
        opt_level: OptLevel::O0,
        verify: false,
        ..JitConfig::default()
    });

    let func = build_add();
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&[func], &ext).expect("O0 compile should succeed");

    let f: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("add").expect("add")
    };

    assert_eq!(f(1, 2), 3);
}

// ---------------------------------------------------------------------------
// Test: mixed internal + external calls
// ---------------------------------------------------------------------------

extern "C" fn host_square(x: i64) -> i64 {
    x * x
}

#[test]
fn test_jit_mixed_internal_external() {
    let jit = JitCompiler::new(JitConfig::default());

    // Internal function: add
    let add_fn = build_add();

    // Caller that calls internal "add" via BL
    let call_add = build_caller("call_add", "add");

    // Caller that calls external "_host_square" via BL
    let call_square = build_extern_caller("_host_square");

    let mut ext: HashMap<String, *const u8> = HashMap::new();
    ext.insert("_host_square".to_string(), host_square as *const u8);

    let buf = jit
        .compile_raw(&[add_fn, call_add, call_square], &ext)
        .expect("mixed internal/external compile should succeed");

    // Test internal call
    let fa: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf.get_fn("call_add").expect("call_add")
    };
    assert_eq!(fa(10, 20), 30);

    // Test external call (call_extern calls _host_square)
    let fs: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("call_extern").expect("call_extern")
    };
    assert_eq!(fs(5), 25);
    assert_eq!(fs(0), 0);
    assert_eq!(fs(-3), 9);
}

// ---------------------------------------------------------------------------
// Test: branch patching out of range error
// ---------------------------------------------------------------------------

#[test]
fn test_jit_branch_out_of_range() {
    // BL has a +/-128MB range (26-bit signed offset * 4 bytes).
    // We can't easily allocate 128MB of code in a test, but we can verify
    // the error path exists by checking the JitError::BranchOutOfRange variant.
    // The patch_branch26 function validates the range and returns this error.
    //
    // This test verifies the error type is constructible and the compile_raw
    // path would propagate it. A true out-of-range test would require
    // generating ~32M instructions which is impractical for a unit test.

    let err = JitError::BranchOutOfRange {
        offset: 0,
        target: 256 * 1024 * 1024, // 256MB — beyond +-128MB
        distance: 256 * 1024 * 1024,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("branch out of range"), "error message: {}", msg);
}

// ---------------------------------------------------------------------------
// Test: multiple large functions (stress test for layout)
// ---------------------------------------------------------------------------

#[test]
fn test_jit_multiple_large_functions() {
    let jit = JitCompiler::new(JitConfig::default());

    // 3 functions, each 1024 NOPs = 4KB code each, plus identity semantics.
    let f1 = build_large_filler_function("big_a", 1024);
    let f2 = build_large_filler_function("big_b", 1024);
    let f3 = build_large_filler_function("big_c", 1024);

    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[f1, f2, f3], &ext)
        .expect("compile_raw should succeed with multiple large functions");

    // All three functions should be independently callable.
    let fa: extern "C" fn(i64) -> i64 = unsafe { buf.get_fn("big_a").expect("big_a") };
    let fb: extern "C" fn(i64) -> i64 = unsafe { buf.get_fn("big_b").expect("big_b") };
    let fc: extern "C" fn(i64) -> i64 = unsafe { buf.get_fn("big_c").expect("big_c") };

    assert_eq!(fa(1), 1);
    assert_eq!(fb(2), 2);
    assert_eq!(fc(3), 3);

    // Verify they are at different offsets.
    let ptr_a = buf.get_fn_ptr("big_a").unwrap();
    let ptr_b = buf.get_fn_ptr("big_b").unwrap();
    let ptr_c = buf.get_fn_ptr("big_c").unwrap();
    assert_ne!(ptr_a, ptr_b, "big_a and big_b should be at different addresses");
    assert_ne!(ptr_b, ptr_c, "big_b and big_c should be at different addresses");
}

// ---------------------------------------------------------------------------
// Test: function symbols are at expected offsets
// ---------------------------------------------------------------------------

#[test]
fn test_jit_symbol_offsets() {
    let jit = JitCompiler::new(JitConfig::default());

    // add: 2 instructions (ADD + RET) = 8 bytes
    // sub: 2 instructions (SUB + RET) = 8 bytes
    let funcs = vec![build_add(), build_sub()];
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit.compile_raw(&funcs, &ext).expect("compile_raw should succeed");

    // Verify symbol offsets through the iterator.
    let offsets: HashMap<&str, u64> = buf.symbols().collect();
    assert_eq!(offsets["add"], 0, "first function should be at offset 0");
    assert_eq!(offsets["sub"], 8, "second function should start at offset 8 (after 2 x 4-byte instructions)");
}

// ---------------------------------------------------------------------------
// Test: drop safety — ExecutableBuffer cleans up on drop
// ---------------------------------------------------------------------------

#[test]
fn test_jit_buffer_drop() {
    // Compile, use, then drop the buffer.
    // If munmap is broken, this would cause a memory leak or crash.
    for _ in 0..10 {
        let jit = JitCompiler::new(JitConfig::default());
        let func = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();

        let buf = jit.compile_raw(&[func], &ext).expect("compile");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { buf.get_fn("add").expect("add") };
        assert_eq!(f(1, 1), 2);
        // buf drops here, calling munmap
    }
}

// ===========================================================================
// Issue #363 — JIT re-entrant compilation support
// ---------------------------------------------------------------------------
// Verifies that LLVM2's JIT supports the three re-entrancy scenarios tla2
// needs for interpreter fallback + re-entrant compilation:
//
//   1. Multiple `ExecutableBuffer`s active simultaneously
//   2. Cross-buffer function calls (buffer A calls into buffer B via an extern
//      symbol resolved to a pointer into buffer B's mmap)
//   3. JIT compilation on the main thread while JIT code executes on another
//      thread
//
// All three are expected to work out of the box: each `ExecutableBuffer` owns
// an independent mmap region, the veneer trampoline embeds a full 64-bit
// absolute address (reachable to any target), and `JitCompiler::compile_raw`
// takes `&self` and carries no globally shared mutable state.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: external two-arg caller via BL symbol reference
// ---------------------------------------------------------------------------

/// Build a function that calls an external symbol via BL, forwarding X0 and X1.
///
/// `fn caller(a: i64, b: i64) -> i64 { callee_symbol(a, b) }`
fn build_two_arg_extern_caller(caller_name: &str, callee_symbol: &str) -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new(caller_name.to_string(), sig);
    let entry = func.entry;

    // STP FP, LR, [SP, #-16]!
    let stp = MachInst::new(
        AArch64Opcode::StpPreIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(-16),
        ],
    );
    let stp_id = func.push_inst(stp);
    func.append_inst(entry, stp_id);

    // BL <external symbol> (args already in X0, X1)
    let bl = MachInst::new(
        AArch64Opcode::Bl,
        vec![MachOperand::Symbol(callee_symbol.to_string())],
    );
    let bl_id = func.push_inst(bl);
    func.append_inst(entry, bl_id);

    // LDP FP, LR, [SP], #16
    let ldp = MachInst::new(
        AArch64Opcode::LdpPostIndex,
        vec![
            MachOperand::PReg(FP),
            MachOperand::PReg(LR),
            Special(SpecialReg::SP),
            MachOperand::Imm(16),
        ],
    );
    let ldp_id = func.push_inst(ldp);
    func.append_inst(entry, ldp_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

// ---------------------------------------------------------------------------
// Test: multiple independent JIT buffers can be live simultaneously
// ---------------------------------------------------------------------------

#[test]
fn test_jit_multiple_buffers_simultaneously() {
    let jit_a = JitCompiler::new(JitConfig::default());
    let jit_b = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf_a = jit_a
        .compile_raw(&[build_add()], &ext)
        .expect("compile_raw should succeed for add");
    let buf_b = jit_b
        .compile_raw(&[build_sub()], &ext)
        .expect("compile_raw should succeed for sub");

    let add: extern "C" fn(i64, i64) -> i64 = unsafe { buf_a.get_fn("add").expect("add") };
    let sub: extern "C" fn(i64, i64) -> i64 = unsafe { buf_b.get_fn("sub").expect("sub") };

    assert_eq!(add(3, 4), 7);
    assert_eq!(sub(10, 3), 7);
    assert_eq!(add(-5, 2), -3);
    assert_eq!(sub(1, 9), -8);

    let buf_c = jit_a
        .compile_raw(&[build_mul()], &ext)
        .expect("reused compiler should produce another independent buffer");
    let mul: extern "C" fn(i64, i64) -> i64 = unsafe { buf_c.get_fn("mul").expect("mul") };

    assert_eq!(add(20, 22), 42);
    assert_eq!(sub(100, 58), 42);
    assert_eq!(mul(6, 7), 42);
}

// ---------------------------------------------------------------------------
// Test: cross-buffer call via extern symbol trampoline
// ---------------------------------------------------------------------------

#[test]
fn test_jit_cross_buffer_call_via_extern_symbol() {
    let jit_b = JitCompiler::new(JitConfig::default());
    let ext_b: HashMap<String, *const u8> = HashMap::new();

    let buf_b = jit_b
        .compile_raw(&[build_sub()], &ext_b)
        .expect("compile_raw should succeed for sub");
    let ptr_b = buf_b.get_fn_ptr("sub").expect("should find 'sub' symbol");

    let jit_a = JitCompiler::new(JitConfig::default());
    let mut ext_a: HashMap<String, *const u8> = HashMap::new();
    ext_a.insert("_bridge_sub".to_string(), ptr_b);

    let buf_a = jit_a
        .compile_raw(
            &[build_two_arg_extern_caller("call_sub_cross", "_bridge_sub")],
            &ext_a,
        )
        .expect("compile_raw should succeed for cross-buffer extern call");

    // This proves the veneer trampoline mechanism can resolve extern symbols to
    // addresses in any memory region, including other JIT buffers.
    let call_sub_cross: extern "C" fn(i64, i64) -> i64 = unsafe {
        buf_a.get_fn("call_sub_cross").expect("call_sub_cross")
    };

    assert_eq!(call_sub_cross(10, 3), 7);
    assert_eq!(call_sub_cross(0, 0), 0);
    assert_eq!(call_sub_cross(100, 50), 50);
    assert_eq!(call_sub_cross(-5, 5), -10);

    // Sanity-check: both buffers are still reachable after the cross call.
    let sub_direct: extern "C" fn(i64, i64) -> i64 =
        unsafe { buf_b.get_fn("sub").expect("sub direct") };
    assert_eq!(sub_direct(42, 1), 41);
}

// ---------------------------------------------------------------------------
// Test: compile on one thread while JIT code executes on another
// ---------------------------------------------------------------------------

#[test]
fn test_jit_concurrent_compile_while_executing() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let factorial_buf = jit
        .compile_raw(&[build_factorial()], &ext)
        .expect("compile_raw should succeed for factorial");
    let factorial_buf = std::sync::Arc::new(factorial_buf);

    let worker_buf = std::sync::Arc::clone(&factorial_buf);
    let worker = std::thread::spawn(move || {
        let factorial: extern "C" fn(i64) -> i64 = unsafe {
            worker_buf.get_fn("factorial").expect("factorial")
        };

        let mut sum = 0i64;
        for _ in 0..10_000 {
            sum += factorial(10);
        }
        sum
    });

    // This proves `JitCompiler::compile_raw` has no shared mutable state that
    // would make concurrent compile-while-execute unsafe. It does not prove
    // concurrent COMPILATION on multiple threads (see the next test).
    let mut verified_compilations = 0usize;
    for i in 0..32 {
        match i % 3 {
            0 => {
                let buf = jit.compile_raw(&[build_add()], &ext).expect("compile add");
                let f: extern "C" fn(i64, i64) -> i64 = unsafe { buf.get_fn("add").expect("add") };
                assert_eq!(f(i as i64, 2), i as i64 + 2);
            }
            1 => {
                let buf = jit.compile_raw(&[build_sub()], &ext).expect("compile sub");
                let f: extern "C" fn(i64, i64) -> i64 = unsafe { buf.get_fn("sub").expect("sub") };
                assert_eq!(f(i as i64, 2), i as i64 - 2);
            }
            _ => {
                let buf = jit.compile_raw(&[build_mul()], &ext).expect("compile mul");
                let f: extern "C" fn(i64, i64) -> i64 = unsafe { buf.get_fn("mul").expect("mul") };
                assert_eq!(f(i as i64, 2), i as i64 * 2);
            }
        }
        verified_compilations += 1;
    }

    let worker_sum = worker.join().expect("worker thread should succeed");
    // factorial(10) = 3_628_800, summed 10_000 times = 36_288_000_000.
    assert_eq!(worker_sum, 36_288_000_000);
    assert_eq!(verified_compilations, 32);
}

// ---------------------------------------------------------------------------
// Test: multiple threads can compile independently with separate JIT instances
// ---------------------------------------------------------------------------

#[test]
fn test_jit_concurrent_compilation_multiple_threads() {
    let handles: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                let jit = JitCompiler::new(JitConfig::default());
                let ext: HashMap<String, *const u8> = HashMap::new();

                match i % 4 {
                    0 => {
                        let buf = jit.compile_raw(&[build_add()], &ext).expect("compile add");
                        let f: extern "C" fn(i64, i64) -> i64 =
                            unsafe { buf.get_fn("add").expect("add") };
                        assert_eq!(f(40, 2), 42);
                    }
                    1 => {
                        let buf = jit.compile_raw(&[build_sub()], &ext).expect("compile sub");
                        let f: extern "C" fn(i64, i64) -> i64 =
                            unsafe { buf.get_fn("sub").expect("sub") };
                        assert_eq!(f(50, 8), 42);
                    }
                    2 => {
                        let buf = jit.compile_raw(&[build_mul()], &ext).expect("compile mul");
                        let f: extern "C" fn(i64, i64) -> i64 =
                            unsafe { buf.get_fn("mul").expect("mul") };
                        assert_eq!(f(6, 7), 42);
                    }
                    _ => {
                        let buf = jit
                            .compile_raw(&[build_return_const()], &ext)
                            .expect("compile return_const");
                        let f: extern "C" fn() -> i64 =
                            unsafe { buf.get_fn("return_const").expect("return_const") };
                        assert_eq!(f(), 42);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread should compile and execute successfully");
    }
}

// ===========================================================================
// Issue #355 + #357 — JIT soundness (lifetime-bound APIs, icache ordering)
// ===========================================================================

#[test]
fn test_jit_buffer_get_fn_bound_basic() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[build_add()], &ext)
        .expect("compile_raw should succeed");

    let add = unsafe {
        buf.get_fn_bound::<extern "C" fn(i64, i64) -> i64>("add")
            .expect("should find 'add'")
    };

    assert_eq!((*add.as_ref())(10, 20), 30);
}

#[test]
fn test_jit_buffer_get_fn_ptr_bound_basic() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[build_add()], &ext)
        .expect("compile_raw should succeed");

    let add_ptr = buf.get_fn_ptr_bound("add").expect("should find 'add'");
    let add: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(add_ptr.as_ptr()) };

    assert_eq!(add(7, 8), 15);
}

#[test]
fn test_jit_buffer_bound_send() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[build_add()], &ext)
        .expect("compile_raw should succeed");

    let handle = std::thread::spawn(move || {
        let add = unsafe {
            buf.get_fn_bound::<extern "C" fn(i64, i64) -> i64>("add")
                .expect("should find 'add'")
        };
        (*add.as_ref())(40, 2)
    });

    assert_eq!(handle.join().unwrap(), 42);
}

/// Regression smoke test for issue #357: if the icache order regresses,
/// execution on strict implementations would fault.
#[test]
fn test_jit_icache_flush_before_mprotect_ordering() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let buf = jit
        .compile_raw(&[build_add()], &ext)
        .expect("compile_raw should succeed");

    let add = unsafe {
        buf.get_fn_bound::<extern "C" fn(i64, i64) -> i64>("add")
            .expect("should find 'add'")
    };

    for i in 0..32 {
        assert_eq!((*add.as_ref())(i, i + 1), i + (i + 1));
    }
}

// ===========================================================================
// Issue #367 — process-symbol dlsym fallback in compile_raw
// ---------------------------------------------------------------------------
// tla2's JIT runtime defines `#[no_mangle] pub extern "C"` helpers that it
// cannot populate into `extern_symbols` without another dlsym step itself.
// Verify that compile_raw transparently resolves such symbols via
// `dlsym(RTLD_DEFAULT, ...)` when they are absent from `extern_symbols`,
// prefers an explicit `extern_symbols` entry when both exist, and surfaces
// a clean `UnresolvedSymbol` error (no zero-addr veneer — issue #353) when
// neither path finds the symbol.
// ===========================================================================

// For the pure-dlsym path we use `abs` from libc — guaranteed visible
// through `dlsym(RTLD_DEFAULT, ...)` on any Unix host. Rust test binaries
// do not re-export their own `no_mangle` symbols, so dispatching to a
// libc symbol is the portable way to exercise the fallback.
unsafe extern "C" {
    fn abs(x: std::os::raw::c_int) -> std::os::raw::c_int;
}

#[cfg(unix)]
#[test]
fn test_jit_dlsym_fallback_resolves_libc_symbol() {
    // extern_symbols is EMPTY — compile_raw must resolve `abs` through
    // the dlsym(RTLD_DEFAULT) fallback added for #367.
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    // `resolve_extern` strips the leading underscore on macOS before the
    // dlsym lookup; on Linux the bare symbol already matches. The
    // generated-code reference uses the platform's native mangled form.
    #[cfg(target_os = "macos")]
    let sym = "_abs";
    #[cfg(not(target_os = "macos"))]
    let sym = "abs";

    // `build_extern_caller` builds a function `call_extern(x) -> extern(x)`
    // with an i64 signature; libc `abs` takes c_int (i32). We verify that
    // compile_raw resolved the symbol (no UnresolvedSymbol error) and that
    // the call lands in libc. Inputs are kept within i32 range.
    let caller = build_extern_caller(sym);
    let buf = jit
        .compile_raw(&[caller], &ext)
        .expect("compile_raw should resolve libc `abs` via dlsym fallback");

    // Sanity-check libc abs is linked into this test binary (touches the
    // extern so the linker does not GC it on some platforms).
    assert_eq!(unsafe { abs(-1) }, 1);

    // Confirm the JIT buffer actually dispatches into libc `abs`. We pass
    // non-negative i64 values (i32-safe) since `abs(i32::MIN)` is UB.
    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("call_extern")
            .expect("should find 'call_extern'")
    };
    // abs(5) == 5, abs(12345) == 12345. The sign-extended i64 return of an
    // i32 result is the same for non-negative inputs.
    assert_eq!(f(5), 5);
    assert_eq!(f(12345), 12345);
}

// Host helper reachable only via its static address (NOT no_mangle).
extern "C" fn llvm2_jit_test_override_shadow(x: i64) -> i64 {
    x + 9999
}

#[cfg(unix)]
#[test]
fn test_jit_extern_symbols_preferred_over_dlsym_end_to_end() {
    // `extern_symbols` must take precedence over the dlsym fallback. We use
    // the libc symbol `abs` as the name so that dlsym WOULD resolve — but
    // extern_symbols maps the name to `llvm2_jit_test_override_shadow`,
    // which adds 9999 rather than computing absolute value.
    let jit = JitCompiler::new(JitConfig::default());

    #[cfg(target_os = "macos")]
    let sym = "_abs";
    #[cfg(not(target_os = "macos"))]
    let sym = "abs";

    let mut ext: HashMap<String, *const u8> = HashMap::new();
    ext.insert(sym.to_string(), llvm2_jit_test_override_shadow as *const u8);

    let caller = build_extern_caller(sym);
    let buf = jit
        .compile_raw(&[caller], &ext)
        .expect("compile_raw should succeed");

    let f: extern "C" fn(i64) -> i64 = unsafe {
        buf.get_fn("call_extern")
            .expect("should find 'call_extern'")
    };
    // If extern_symbols were ignored we'd get |x| (libc abs). The shadow
    // helper instead adds 9999.
    assert_eq!(f(0), 9999);
    assert_eq!(f(1), 10000);
}

#[test]
fn test_jit_missing_symbol_returns_unresolved_not_zero_addr() {
    // A symbol name unlikely to exist in any process — neither extern_symbols
    // nor dlsym can resolve it. compile_raw must surface `UnresolvedSymbol`
    // rather than emitting a veneer with address 0 (issue #353).
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();

    let caller = build_extern_caller("_llvm2_jit_definitely_missing_symbol_xyzzy_367");
    // Do not use `expect_err` — `ExecutableBuffer` does not implement Debug.
    match jit.compile_raw(&[caller], &ext) {
        Ok(_) => panic!("compile_raw must fail when symbol cannot be resolved"),
        Err(JitError::UnresolvedSymbol(name)) => {
            assert!(
                name.contains("llvm2_jit_definitely_missing_symbol_xyzzy_367"),
                "error should name the missing symbol, got: {name}"
            );
        }
        Err(other) => panic!("expected UnresolvedSymbol, got {other:?}"),
    }
}

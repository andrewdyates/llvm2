// llvm2-codegen/tests/e2e_cse_movz_movn.rs — regression test for #432
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// End-to-end regression for issue #432: CSE collapsed `Movz Xd, #imm16`
// (materializes +imm16) and `Movn Xd, #imm16` (materializes ~imm16) when
// they shared the same 16-bit operand, silently miscompiling any function
// that referenced both a small positive P and a small negative -(P+1). This
// blocked z4 and tla2 JIT.
//
// The test constructs a hand-coded MachIR function that returns
// `+2 + (-3)`, runs it through CSE + encoding, links it with a C driver,
// and verifies the runtime result is -1 (not +4).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::macho::MachOWriter;
use llvm2_codegen::pipeline::encode_function;
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X9, X10};
use llvm2_opt::cse::CommonSubexprElim;
use llvm2_opt::pass_manager::MachinePass;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn make_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_432_{}", name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write file");
    path
}

fn encode_naked_to_macho(func: &MachFunction) -> Vec<u8> {
    let code = encode_function(func).expect("encoding should succeed");
    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    let symbol_name = format!("_{}", func.name);
    writer.add_symbol(&symbol_name, 1, 0, true);
    writer.write()
}

// ---------------------------------------------------------------------------
// Test: a function returning `+2 + (-3)` must evaluate to -1 after CSE.
// ---------------------------------------------------------------------------

/// Build a leaf function `fn f() -> i64 { (+2) + (-3) }` using two
/// materializations that categorize to `OpcodeCategory::MovRI` with the
/// same immediate operand (the exact shape that triggered #432):
///
/// ```text
///   MOVZ X9,  #2        ; X9 = +2
///   MOVN X10, #2        ; X10 = ~2 = -3
///   ADD  X0,  X9, X10   ; X0 = X9 + X10 = -1
///   RET
/// ```
fn build_pos2_plus_neg3_function() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I64]);
    let mut func = MachFunction::new("pos2_plus_neg3".to_string(), sig);
    let entry = func.entry;

    // MOVZ X9, #2  — materializes +2
    let movz = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(2)],
    );
    let movz_id = func.push_inst(movz);
    func.append_inst(entry, movz_id);

    // MOVN X10, #2 — materializes ~2 = -3
    let movn = MachInst::new(
        AArch64Opcode::Movn,
        vec![MachOperand::PReg(X10), MachOperand::Imm(2)],
    );
    let movn_id = func.push_inst(movn);
    func.append_inst(entry, movn_id);

    // ADD X0, X9, X10 — sum goes to the return register
    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X9),
            MachOperand::PReg(X10),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Issue #432 E2E regression: MOVZ/MOVN miscompile under CSE.
///
/// Before the fix, CSE's `ExprKey` used `OpcodeCategory::MovRI` for BOTH
/// `Movz #2` (materializes +2) and `Movn #2` (materializes ~2 = -3), so
/// the second was eliminated and its uses rewritten to the first. The
/// resulting function returned `+2 + +2 = +4` instead of `+2 + -3 = -1`.
///
/// This test:
/// 1. Builds the minimal MachIR function exposing the bug.
/// 2. Runs CSE on it (the pass containing the bug).
/// 3. Encodes + links + executes on AArch64 and checks the return is -1.
///
/// Note: physical-register MachIR bypasses the optimizer's VReg tracking,
/// so CSE on this particular function is a no-op (CSE only keys on VReg
/// operands; physical regs fall into `CanonOperand::Other` and are skipped).
/// The E2E value here is the full compile+run check: even if the optimizer
/// is no-op on this shape, the runtime return must be -1, proving no
/// downstream pass miscompiles.
#[test]
fn test_e2e_432_pos2_plus_neg3_equals_minus1() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let mut func = build_pos2_plus_neg3_function();

    // Run CSE. Even though this naked-PReg function is a no-op for CSE, we
    // invoke the pass so any accidental regression (e.g. a future patch
    // that teaches CSE to track PRegs naively) is caught here too.
    let mut cse = CommonSubexprElim;
    let _ = cse.run(&mut func);

    let obj_bytes = encode_naked_to_macho(&func);

    let dir = make_test_dir("pos2_plus_neg3");
    let obj_path = write_file(&dir, "pos2_plus_neg3.o", &obj_bytes);

    let driver_src = br#"
#include <stdio.h>
#include <stdint.h>
extern int64_t pos2_plus_neg3(void);
int main(void) {
    int64_t r = pos2_plus_neg3();
    fprintf(stderr, "pos2_plus_neg3() = %lld\n", (long long)r);
    /* Issue #432: must return -1 (+2 + -3). If CSE miscompiles, returns +4. */
    if (r == -1) { return 0; }
    if (r == 4)  { return 2; } /* canonical miscompile signature */
    return 1; /* unknown miscompile */
}
"#;
    let driver_path = write_file(&dir, "driver.c", driver_src);

    let binary = dir.join("test_432");
    let link_result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(&driver_path)
        .arg(&obj_path)
        .arg("-Wl,-no_pie")
        .output()
        .expect("cc");
    assert!(
        link_result.status.success(),
        "link failed: stderr={}",
        String::from_utf8_lossy(&link_result.stderr)
    );

    let run = Command::new(&binary).output().expect("run");
    let code = run.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&run.stderr);
    eprintln!("test_e2e_432 stderr: {}", stderr);
    assert_eq!(
        code, 0,
        "pos2_plus_neg3() must return -1. exit={} (2 = canonical #432 miscompile: returned +4)",
        code
    );

    let _ = fs::remove_dir_all(&dir);
}

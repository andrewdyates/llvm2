// Smallest no_std/`main`-only program for WS4 M0 smoke testing.
//
// This file is fed to rustc under `-Zcodegen-backend=<llvm2-dylib>`. At
// M0 we EXPECT this compile to fail with our backend's fatal
// diagnostic, proving that rustc successfully loaded the dylib and
// reached our `codegen_crate` entrypoint. When M1 lands, the same
// input will start producing a runnable binary.
fn main() {
    loop {}
}

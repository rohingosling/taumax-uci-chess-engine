//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Integration tests for the UCI identity and readiness handshake.
//
//----------------------------------------------------------------------------------------------------------------------

use std::io::Write;
use std::process::{Command, Stdio};

use assert_cmd::prelude::*;
use predicates::prelude::*;

//----------------------------------------------------------------------------------------------------------------------
// Function: run_engine_script
//
// Description:
//
//   Run the engine binary with a scripted UCI command sequence.
//
//----------------------------------------------------------------------------------------------------------------------

fn run_engine_script(script: &str) -> String {

    // Integration tests exercise the compiled binary, not just library functions. That catches basic
    // process-level UCI behavior: stdin commands in, stdout protocol lines out.

    let mut child = Command::cargo_bin("taumax")
        .unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Taking stdin moves the pipe handle out of child. Dropping it after writing sends EOF, which lets
    // the engine finish if the script has also sent "quit".

    let mut standard_input = child.stdin.take().unwrap();
    standard_input.write_all(script.as_bytes()).unwrap();
    drop(standard_input);

    let output = child.wait_with_output().unwrap();

    // If the child exits unsuccessfully, include stderr in the assertion message because protocol
    // diagnostics are intentionally written there.

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: uci_handshake_returns_identity_and_uciok
//
// Description:
//
//   Verify the uci command returns identity lines and uciok.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn uci_handshake_returns_identity_and_uciok() {

    // Run the startup handshake sequence that a GUI sends immediately after launching the engine.

    let output = run_engine_script("uci\nquit\n");

    // Verify identity and completion lines are present in stdout.

    assert!(predicate::str::contains("id name Taumax").eval(&output));
    assert!(predicate::str::contains("id author Rohin Gosling").eval(&output));
    assert!(predicate::str::contains("uciok").eval(&output));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: isready_returns_readyok
//
// Description:
//
//   Verify the isready command returns readyok.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn isready_returns_readyok() {

    // Ask the engine for readiness without performing a full handshake first.

    let output = run_engine_script("isready\nquit\n");

    // The engine should answer the UCI synchronization token.

    assert!(predicate::str::contains("readyok").eval(&output));
}

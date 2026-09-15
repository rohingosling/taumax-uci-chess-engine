//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Validation-position checks for the active selector surface.
//
//----------------------------------------------------------------------------------------------------------------------

use std::io::Write;
use std::process::{Command, Stdio};

use assert_cmd::prelude::*;

//----------------------------------------------------------------------------------------------------------------------
// Struct: ValidationPosition
//
// Description:
//
//   Stores one validation position.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct ValidationPosition {
    name: &'static str,
    position_command: &'static str,
    expected_best_move: &'static str,
}

const VALIDATION_POSITIONS: [ValidationPosition; 3] = [
    ValidationPosition {
        name: "starting position",
        position_command: "position startpos",
        expected_best_move: "a2a4",
    },
    ValidationPosition {
        name: "queen bait",
        position_command: "position fen 4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
        expected_best_move: "e1e2",
    },
    ValidationPosition {
        name: "sparse king endgame",
        position_command: "position fen 8/8/8/8/8/8/4K3/6k1 w - - 0 1",
        expected_best_move: "e2d1",
    },
];

//----------------------------------------------------------------------------------------------------------------------
// Function: run_engine_script
//
// Description:
//
//   Run the engine binary with a scripted UCI command sequence.
//
//----------------------------------------------------------------------------------------------------------------------

fn run_engine_script(script: &str) -> String {

    // Launch the real engine binary so deterministic strategy checks include process-level I/O.

    let mut child = Command::cargo_bin("taumax")
        .unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Send the complete UCI script and close stdin so the engine can finish after quit.

    let mut standard_input = child.stdin.take().unwrap();
    standard_input.write_all(script.as_bytes()).unwrap();
    drop(standard_input);

    // Wait for the process and include stderr in the assertion when execution fails.

    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Return the engine's protocol stdout for bestmove extraction.

    String::from_utf8(output.stdout).unwrap()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: best_move_from_output
//
// Description:
//
//   Extract the UCI bestmove value from engine stdout.
//
//----------------------------------------------------------------------------------------------------------------------

fn best_move_from_output(output: &str) -> &str {

    // Extract the move text from the single bestmove response in the script output.

    output
        .lines()
        .find_map(|line| line.strip_prefix("bestmove "))
        .expect("missing bestmove line")
}

//----------------------------------------------------------------------------------------------------------------------
// Function: random_validation_positions_remain_reproducible
//
// Description:
//
//   Verify the active seeded random control condition stays reproducible.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn random_validation_positions_remain_reproducible() {

    // Iterate through fixed positions whose expected Random selections are tied to the seed.

    for position in VALIDATION_POSITIONS {

        // Build a short script that fixes Strategy, Random Seed, board position, and depth.

        let script = format!(
            "setoption name Strategy value Random\n\
             setoption name Random Seed value random validation\n\
             {}\n\
             go depth 2\n\
             quit\n",
            position.position_command
        );

        // Run the script and compare the selected move with the stored validation expectation.

        let output = run_engine_script(&script);
        let best_move = best_move_from_output(&output);

        assert_eq!(
            best_move, position.expected_best_move,
            "Random on {}",
            position.name
        );
    }
}

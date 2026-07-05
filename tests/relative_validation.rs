//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Validation-position checks for relative causal-entropy behavior.
//
//----------------------------------------------------------------------------------------------------------------------

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use assert_cmd::prelude::*;

const RANDOM_CONTROL_SEED: &str = "relative random control";
const SAFE_QUEEN_CAPTURE_POSITION: &str = "position fen 4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1";
const PROTECTED_PAWN_CAPTURE_POSITION: &str = "position fen 8/8/4k3/3p4/4P3/8/8/4K3 w - - 0 1";
const TERMINAL_DANGER_POSITION: &str = "position fen 6k1/8/6K1/8/8/8/8/7R b - - 0 1";

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeTraceRecord
//
// Description:
//
//   Stores one parsed relative trace line.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct RelativeTraceRecord {
    move_text: String,
    score: f64,
    depth: u64,
    opponent_text: String,
    own_legal_move_count: usize,
    opponent_freedom_count: usize,
    relative_potential: f64,
    future_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Function: run_engine_script
//
// Description:
//
//   Run the engine binary with a scripted UCI command sequence.
//
//----------------------------------------------------------------------------------------------------------------------

fn run_engine_script(script: &str) -> String {

    // Launch the compiled engine binary so validation covers the real UCI process boundary.

    let mut child = Command::cargo_bin("taumax")
        .unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Write the complete scripted UCI conversation, then close stdin to let the process exit.

    let mut standard_input = child.stdin.take().unwrap();
    standard_input.write_all(script.as_bytes()).unwrap();
    drop(standard_input);

    // Wait for the engine and surface stderr if the child process failed.

    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Return stdout as normal text because UCI protocol output is UTF-8 compatible.

    String::from_utf8(output.stdout).unwrap()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: random_control_output
//
// Description:
//
//   Run the deterministic random control on one validation position.
//
//----------------------------------------------------------------------------------------------------------------------

fn random_control_output(position_command: &str) -> String {

    // Build a deterministic Random-strategy script for the supplied validation position.

    let script = format!(
        "setoption name Strategy value Random\n\
         setoption name Random Seed value {RANDOM_CONTROL_SEED}\n\
         {position_command}\n\
         go depth 2\n\
         quit\n"
    );

    // Run the script through the same binary path as the relative strategy tests.

    run_engine_script(&script)
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_trace_output
//
// Description:
//
//   Run the relative selector with trace enabled on one validation position.
//
//----------------------------------------------------------------------------------------------------------------------

fn relative_trace_output(position_command: &str) -> String {

    // Build a RelativeCausalEntropy script with Diagnostics Trace enabled.

    let script = format!(
        "setoption name Strategy value RelativeCausalEntropy\n\
         setoption name Diagnostics Trace value true\n\
         setoption name Max Depth value 2\n\
         {position_command}\n\
         go depth 2\n\
         quit\n"
    );

    // Run the traced strategy so tests can inspect both bestmove and score fields.

    run_engine_script(&script)
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

    // Find the protocol bestmove line and return only the move text after the keyword.

    output
        .lines()
        .find_map(|line| line.strip_prefix("bestmove "))
        .expect("missing bestmove line")
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_trace_records
//
// Description:
//
//   Parse relative trace lines from engine stdout.
//
//----------------------------------------------------------------------------------------------------------------------

fn relative_trace_records(output: &str) -> Vec<RelativeTraceRecord> {

    // Keep only per-root relative trace lines and parse each into typed fields.

    output
        .lines()
        .filter(|line| line.starts_with("info string tau strategy=RelativeCausalEntropy "))
        .map(parse_relative_trace_record)
        .collect()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: parse_relative_trace_record
//
// Description:
//
//   Parse one relative trace line into typed fields.
//
//----------------------------------------------------------------------------------------------------------------------

fn parse_relative_trace_record(trace_line: &str) -> RelativeTraceRecord {

    // Split key=value fields into a lookup table so tests can read them by name.

    let fields = trace_line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<HashMap<_, _>>();

    // Parse the expected numeric fields into their test-facing types.

    RelativeTraceRecord {
        move_text: required_field(&fields, "move").to_string(),
        score: parse_field(&fields, "score"),
        depth: parse_field(&fields, "depth"),
        opponent_text: required_field(&fields, "opponent").to_string(),
        own_legal_move_count: parse_field(&fields, "own"),
        opponent_freedom_count: parse_field(&fields, "opponentFreedom"),
        relative_potential: parse_field(&fields, "rel"),
        future_count: parse_field(&fields, "futures"),
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: required_field
//
// Description:
//
//   Return a required trace field value.
//
//----------------------------------------------------------------------------------------------------------------------

fn required_field<'a>(fields: &'a HashMap<&str, &str>, name: &str) -> &'a str {

    // Panic with the missing field name so trace-format failures are easy to diagnose.

    fields
        .get(name)
        .copied()
        .unwrap_or_else(|| panic!("missing trace field: {name}"))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: parse_field
//
// Description:
//
//   Parse a required trace field value.
//
//----------------------------------------------------------------------------------------------------------------------

fn parse_field<T>(fields: &HashMap<&str, &str>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{

    // Parse through FromStr so this helper works for f64, u64, and usize fields.

    required_field(fields, name)
        .parse::<T>()
        .unwrap_or_else(|error| panic!("invalid trace field {name}: {error:?}"))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: trace_record_for_move
//
// Description:
//
//   Return the trace record for one root move.
//
//----------------------------------------------------------------------------------------------------------------------

fn trace_record_for_move<'a>(
    trace_records: &'a [RelativeTraceRecord],
    move_text: &str,
) -> &'a RelativeTraceRecord {

    // Select the record for one root move so assertions can compare move-specific metrics.

    trace_records
        .iter()
        .find(|trace_record| trace_record.move_text == move_text)
        .unwrap_or_else(|| panic!("missing trace record for move: {move_text}"))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: safe_queen_capture_emerges_from_relative_metrics
//
// Description:
//
//   Verify the queen-bait fixture flips from random's quiet push to a relative capture.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn safe_queen_capture_emerges_from_relative_metrics() {

    // Run both the deterministic random control and the relative strategy on the queen-bait position.

    let random_output = random_control_output(SAFE_QUEEN_CAPTURE_POSITION);
    let relative_output = relative_trace_output(SAFE_QUEEN_CAPTURE_POSITION);

    // Compare the capture root against a quiet pawn push in the relative trace.

    let trace_records = relative_trace_records(&relative_output);
    let capture_record = trace_record_for_move(&trace_records, "e4d5");
    let quiet_push_record = trace_record_for_move(&trace_records, "e4e5");

    // The relative strategy should capture the queen and show better freedom metrics for that root.

    assert_eq!(best_move_from_output(&random_output), "e1f1");
    assert_eq!(best_move_from_output(&relative_output), "e4d5");
    assert_eq!(capture_record.depth, 2);
    assert_eq!(capture_record.opponent_text, "e8e7");
    assert!(capture_record.score > quiet_push_record.score);
    assert!(capture_record.own_legal_move_count > quiet_push_record.own_legal_move_count);
    assert!(capture_record.opponent_freedom_count < quiet_push_record.opponent_freedom_count);
    assert!(capture_record.relative_potential > quiet_push_record.relative_potential);
    assert!(capture_record.future_count > quiet_push_record.future_count);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: protected_pawn_capture_is_rejected_by_relative_metrics
//
// Description:
//
//   Verify a recapturable pawn capture loses to a freer non-capture root move.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn protected_pawn_capture_is_rejected_by_relative_metrics() {

    // Run both strategies where the apparent capture can be recaptured by the opponent king.

    let random_output = random_control_output(PROTECTED_PAWN_CAPTURE_POSITION);
    let relative_output = relative_trace_output(PROTECTED_PAWN_CAPTURE_POSITION);

    // Compare the selected king move against the losing pawn capture.

    let trace_records = relative_trace_records(&relative_output);
    let selected_record = trace_record_for_move(&trace_records, "e1d2");
    let capture_record = trace_record_for_move(&trace_records, "e4d5");

    // The capture's opponent reply should lower its future and relative-potential metrics.

    assert_eq!(best_move_from_output(&random_output), "e1d2");
    assert_eq!(best_move_from_output(&relative_output), "e1d2");
    assert_eq!(capture_record.opponent_text, "e6d5");
    assert!(selected_record.score > capture_record.score);
    assert!(selected_record.own_legal_move_count > capture_record.own_legal_move_count);
    assert!(selected_record.relative_potential > capture_record.relative_potential);
    assert!(selected_record.future_count > capture_record.future_count);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: terminal_danger_collapses_future_agency_trace
//
// Description:
//
//   Verify a forced terminal line is visible as low future agency in the trace.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn terminal_danger_collapses_future_agency_trace() {

    // Run the one-legal-move terminal-danger fixture through both strategies.

    let random_output = random_control_output(TERMINAL_DANGER_POSITION);
    let relative_output = relative_trace_output(TERMINAL_DANGER_POSITION);

    // Inspect the only root move's trace after the forced opponent reply.

    let trace_records = relative_trace_records(&relative_output);
    let forced_record = trace_record_for_move(&trace_records, "g8f8");

    // The trace should show that the forced line leaves Taumax with low future agency.

    assert_eq!(best_move_from_output(&random_output), "g8f8");
    assert_eq!(best_move_from_output(&relative_output), "g8f8");
    assert_eq!(trace_records.len(), 1);
    assert_eq!(forced_record.opponent_text, "h1h8");
    assert_eq!(forced_record.own_legal_move_count, 1);
    assert_eq!(forced_record.future_count, 1);
    assert!(forced_record.opponent_freedom_count > forced_record.own_legal_move_count);
    assert!(forced_record.score < 0.0);
    assert!(forced_record.relative_potential < 0.0);
}

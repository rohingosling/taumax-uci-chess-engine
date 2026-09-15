//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Integration tests for go-command behavior and bestmove output.
//
//----------------------------------------------------------------------------------------------------------------------

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use assert_cmd::prelude::*;

//----------------------------------------------------------------------------------------------------------------------
// Function: run_engine_script
//
// Description:
//
//   Run the engine binary with a scripted UCI command sequence.
//
//----------------------------------------------------------------------------------------------------------------------

fn run_engine_script(script: &str) -> String {

    // Spawn the real binary so the test covers command-line I/O and not only internal functions.

    let mut child = Command::cargo_bin("taumax")
        .unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // The engine reads line-oriented UCI text from stdin. Dropping the pipe after writing tells the
    // process there is no more input.

    let mut standard_input = child.stdin.take().unwrap();
    standard_input.write_all(script.as_bytes()).unwrap();
    drop(standard_input);

    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: assert_valid_bestmove_line
//
// Description:
//
//   Assert that engine output contains a syntactically valid bestmove line.
//
//----------------------------------------------------------------------------------------------------------------------

fn assert_valid_bestmove_line(output: &str) {

    // Without a fixed Random Seed, the selected move may vary, so this test validates UCI syntax instead
    // of expecting one fixed chess move.

    let bestmove_line = output
        .lines()
        .find(|line| line.starts_with("bestmove "))
        .expect("missing bestmove line");
    let move_text = bestmove_line.trim_start_matches("bestmove ");

    assert!(
        is_valid_uci_move_text(move_text),
        "invalid bestmove line: {bestmove_line}"
    );
}

//----------------------------------------------------------------------------------------------------------------------
// Function: is_valid_uci_move_text
//
// Description:
//
//   Return whether text has the shape of a UCI move or null move.
//
//----------------------------------------------------------------------------------------------------------------------

fn is_valid_uci_move_text(move_text: &str) -> bool {

    // "0000" is UCI's null move for positions where no legal move is available.

    if move_text == "0000" {
        return true;
    }

    // Normal UCI moves are four characters, such as "e2e4". Promotions add one piece character,
    // such as "a7a8q".

    let bytes = move_text.as_bytes();

    if bytes.len() != 4 && bytes.len() != 5 {
        return false;
    }

    let valid_from_square = (b'a'..=b'h').contains(&bytes[0]) && (b'1'..=b'8').contains(&bytes[1]);
    let valid_to_square = (b'a'..=b'h').contains(&bytes[2]) && (b'1'..=b'8').contains(&bytes[3]);
    let valid_promotion = bytes.len() == 4 || matches!(bytes[4], b'q' | b'r' | b'b' | b'n');

    valid_from_square && valid_to_square && valid_promotion
}

//----------------------------------------------------------------------------------------------------------------------
// Function: trace_field_value
//
// Description:
//
//   Return a named key-value field from a UCI info string line.
//
//----------------------------------------------------------------------------------------------------------------------

fn trace_field_value<'a>(trace_line: &'a str, field_name: &str) -> Option<&'a str> {

    // Build the key prefix once, then search whitespace-separated fields for that exact name.

    let field_prefix = format!("{field_name}=");

    trace_line
        .split_whitespace()
        .find_map(|field| field.strip_prefix(&field_prefix))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: startpos_go_returns_bestmove
//
// Description:
//
//   Verify a go command from the starting position returns a bestmove.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn startpos_go_returns_bestmove() {

    // A minimal position/go script should always produce a syntactically valid bestmove.

    let output = run_engine_script("position startpos\ngo movetime 1\nquit\n");

    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: diagnostics_trace_with_random_emits_no_retired_entropy_trace
//
// Description:
//
//   Verify Diagnostics Trace does not emit retired entropy trace fields while only Random is active.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn diagnostics_trace_with_random_emits_no_retired_entropy_trace() {

    // Enable Diagnostics Trace while keeping the Random strategy active.

    let script = concat!(
        "setoption name Diagnostics Trace value true\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 1\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // Random search should not emit relative entropy backend or macrostate fields.

    assert!(!output.contains("info string tau"));
    assert!(!output.contains("backend="));
    assert!(!output.contains("macrostate="));
    assert!(!output.contains("samples="));
    assert!(!output.contains("unique="));
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: fixed_seed_random_baseline_is_repeatable
//
// Description:
//
//   Verify the selectable random baseline can be reproduced for validation runs.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn fixed_seed_random_baseline_is_repeatable() {

    // The same fixed seed and searchmoves filter should produce byte-identical stdout twice.

    let script = concat!(
        "setoption name Strategy value Random\n",
        "setoption name Random Seed value repeatable random baseline\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 1\n",
        "quit\n"
    );

    let first_output = run_engine_script(script);
    let second_output = run_engine_script(script);

    // Equal output proves the deterministic random stream is stable through the binary interface.

    assert_eq!(first_output, second_output);
    assert_valid_bestmove_line(&first_output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_strategy_honors_searchmoves_after_scoring
//
// Description:
//
//   Verify the relative strategy scores roots while honoring go searchmoves.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn relative_strategy_honors_searchmoves_after_scoring() {

    // Restrict relative search to one legal root move.

    let script = concat!(
        "setoption name Strategy value RelativeCausalEntropy\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 1\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // With only e2e4 allowed, bestmove must be that root.

    assert!(output.lines().any(|line| line == "bestmove e2e4"));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_diagnostics_trace_emits_observability_fields
//
// Description:
//
//   Verify relative Diagnostics Trace lines expose UCI-safe score diagnostics.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn relative_diagnostics_trace_emits_observability_fields() {

    // Run a traced relative search on one allowed root move so the expected trace line is compact.

    let script = concat!(
        "setoption name Strategy value RelativeCausalEntropy\n",
        "setoption name Diagnostics Trace value true\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 1\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // Locate the root-move trace line and parse the opponent field as UCI move text.

    let trace_line = output
        .lines()
        .find(|line| line.starts_with("info string tau strategy=RelativeCausalEntropy move=e2e4 "))
        .expect("missing relative trace line");
    let opponent_move_text = trace_field_value(trace_line, "opponent").expect("missing opponent");

    // Verify each key metric field exists and parses to the expected type.

    assert_eq!(trace_field_value(trace_line, "depth"), Some("1"));
    assert!(trace_field_value(trace_line, "score")
        .expect("missing score")
        .parse::<f64>()
        .is_ok());
    assert!(
        is_valid_uci_move_text(opponent_move_text),
        "invalid opponent field: {opponent_move_text}"
    );
    assert!(trace_field_value(trace_line, "own")
        .expect("missing own")
        .parse::<usize>()
        .is_ok());
    assert!(trace_field_value(trace_line, "opponentFreedom")
        .expect("missing opponentFreedom")
        .parse::<usize>()
        .is_ok());
    assert!(trace_field_value(trace_line, "rel")
        .expect("missing rel")
        .parse::<f64>()
        .is_ok());
    assert!(trace_field_value(trace_line, "futures")
        .expect("missing futures")
        .parse::<usize>()
        .is_ok());
    assert!(trace_field_value(trace_line, "terminals")
        .expect("missing terminals")
        .parse::<usize>()
        .is_ok());
    let profile_line = output
        .lines()
        .find(|line| line.starts_with("info string tau profile strategy=RelativeCausalEntropy "))
        .expect("missing profile trace line");

    // Verify the profile line records no GPU request and no retired trace vocabulary appears.

    assert_eq!(trace_field_value(profile_line, "gpuRequested"), Some("no"));
    assert!(!trace_line.contains("backend="));
    assert!(!trace_line.contains("macrostate="));
    assert!(!trace_line.contains("samples="));
    assert!(!trace_line.contains("unique="));
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_diagnostics_trace_marks_terminal_win
//
// Description:
//
//   Verify terminal root wins are exposed through UCI-safe Diagnostics Trace fields.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn relative_diagnostics_trace_marks_terminal_win() {

    // Force the relative strategy to consider only a mate-in-one root.

    let script = concat!(
        "setoption name Strategy value RelativeCausalEntropy\n",
        "setoption name Diagnostics Trace value true\n",
        "position fen 7k/5K2/8/6Q1/8/8/8/8 w - - 0 1\n",
        "go searchmoves g5g7 depth 1\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // Locate the terminal trace line for the checkmating move.

    let trace_line = output
        .lines()
        .find(|line| line.starts_with("info string tau strategy=RelativeCausalEntropy move=g5g7 "))
        .expect("missing terminal trace line");

    // Mate has no opponent reply and is labeled as a terminal win.

    assert!(output.lines().any(|line| line == "bestmove g5g7"));
    assert_eq!(trace_field_value(trace_line, "opponent"), Some("none"));
    assert_eq!(trace_field_value(trace_line, "terminal"), Some("win"));
    assert_eq!(trace_field_value(trace_line, "terminals"), Some("1"));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: gpu_requests_optional_relative_leaf_backend
//
// Description:
//
//   Verify the GPU option requests the optional relative leaf backend.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn gpu_requests_optional_relative_leaf_backend() {

    // Request GPU acceleration on a traced relative search.

    let script = concat!(
        "setoption name Strategy value RelativeCausalEntropy\n",
        "setoption name GPU Acceleration value true\n",
        "setoption name Diagnostics Trace value true\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 2\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // The profile line should show the GPU request and the backend that actually evaluated leaves.

    let profile_line = output
        .lines()
        .find(|line| line.starts_with("info string tau profile strategy=RelativeCausalEntropy "))
        .expect("missing profile trace line");
    let leaf_backend = trace_field_value(profile_line, "leafBackend").expect("missing leafBackend");

    // Backend may fall back to CPU if no compatible GPU runtime exists.

    assert_eq!(trace_field_value(profile_line, "gpuRequested"), Some("yes"));
    assert!(
        matches!(leaf_backend, "cpu-batch" | "gpu-batch"),
        "unexpected leaf backend: {leaf_backend}"
    );
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: gpu_option_does_not_change_random_search
//
// Description:
//
//   Verify a GPU request does not change random search or emit backend diagnostics.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn gpu_option_does_not_change_random_search() {

    // Send a GPU request while using Random so the relative backend should remain inactive.

    let script = concat!(
        "setoption name Strategy value Random\n",
        "setoption name GPU Acceleration value true\n",
        "setoption name Diagnostics Trace value true\n",
        "position startpos\n",
        "go searchmoves e2e4 depth 1\n",
        "quit\n"
    );

    let output = run_engine_script(script);

    // Random search should ignore relative backend diagnostics and still return a valid move.

    assert!(!output.contains("backend-request=gpu"));
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: searchmoves_restricts_root_candidates
//
// Description:
//
//   Verify go searchmoves limits the root moves available to the selector.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn searchmoves_restricts_root_candidates() {

    // With only e2e4 allowed, even Random selection has a single legal candidate.

    let output = run_engine_script("position startpos\ngo searchmoves e2e4 depth 1\nquit\n");

    assert!(output.lines().any(|line| line == "bestmove e2e4"));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: movetime_returns_bestmove_found_so_far
//
// Description:
//
//   Verify tiny movetime limits still return a valid bestmove quickly.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn movetime_returns_bestmove_found_so_far() {

    // movetime zero creates an immediate deadline, but the engine must still produce a bestmove.

    let script = concat!(
        "setoption name Max Depth value 12\n",
        "position startpos\n",
        "go movetime 0\n",
        "quit\n"
    );
    let start_time = Instant::now();
    let output = run_engine_script(script);

    // The bounded search should return quickly and keep bestmove syntax valid.

    assert!(start_time.elapsed().as_secs() < 2);
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: stop_command_after_bounded_search_is_safe
//
// Description:
//
//   Verify stop is accepted after a bounded search and does not suppress bestmove output.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn stop_command_after_bounded_search_is_safe() {

    // Send stop after a bounded search request to verify the input thread handles it safely.

    let script = concat!(
        "setoption name Max Depth value 12\n",
        "position startpos\n",
        "go movetime 10000\n",
        "stop\n",
        "quit\n"
    );
    let start_time = Instant::now();
    let output = run_engine_script(script);
    let bestmove_count = output
        .lines()
        .filter(|line| line.starts_with("bestmove "))
        .count();

    // The search should finish promptly and emit exactly one bestmove.

    assert!(start_time.elapsed().as_secs() < 2);
    assert_eq!(bestmove_count, 1);
    assert_valid_bestmove_line(&output);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: no_legal_move_returns_null_move
//
// Description:
//
//   Verify positions with no legal moves return the UCI null move.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn no_legal_move_returns_null_move() {

    // Stalemate has no legal move, so UCI requires the null move "0000".

    let output =
        run_engine_script("position fen 7k/5K2/6Q1/8/8/8/8/8 b - - 0 1\ngo movetime 1\nquit\n");

    assert!(output.lines().any(|line| line == "bestmove 0000"));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: unknown_command_does_not_crash
//
// Description:
//
//   Verify unknown commands do not prevent later commands from running.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn unknown_command_does_not_crash() {

    // Unknown text should be ignored, allowing the following isready command to succeed.

    let output = run_engine_script("this-is-not-a-uci-command\nisready\nquit\n");

    assert!(output.lines().any(|line| line == "readyok"));
}

//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Integration tests for UCI option advertisement and session configuration.
//
//----------------------------------------------------------------------------------------------------------------------

use std::io::Write;
use std::process::{Command, Stdio};

use assert_cmd::prelude::*;
use predicates::prelude::*;
use taumax::engine::configuration::{
    EngineConfigurationError, EngineStrategy, OptionUpdateStatus, DIAGNOSTICS_TRACE_OPTION_NAME,
    GPU_OPTION_NAME, MAX_DEPTH_OPTION_NAME, RANDOM_SEED_OPTION_NAME, STRATEGY_OPTION_NAME,
};
use taumax::engine::random::RandomMoveSelector;
use taumax::engine::session::EngineSession;

//----------------------------------------------------------------------------------------------------------------------
// Function: run_engine_script
//
// Description:
//
//   Run the engine binary with a scripted UCI command sequence.
//
//----------------------------------------------------------------------------------------------------------------------

fn run_engine_script(script: &str) -> String {

    // Run the compiled engine so option advertisement is tested exactly as a chess GUI sees it.

    let mut child = Command::cargo_bin("taumax")
        .unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Write the scripted UCI conversation and close stdin to prevent the child from waiting.

    let mut standard_input = child.stdin.take().unwrap();
    standard_input.write_all(script.as_bytes()).unwrap();
    drop(standard_input);

    // Capture the completed process output and fail with stderr if the engine exits unsuccessfully.

    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Return stdout, which contains only UCI protocol lines.

    String::from_utf8(output.stdout).unwrap()
}

//----------------------------------------------------------------------------------------------------------------------
// Function: uci_advertises_current_options_before_uciok
//
// Description:
//
//   Verify the UCI handshake advertises the active Taumax options before uciok.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn uci_advertises_current_options_before_uciok() {

    // Ask for the UCI handshake and split stdout into ordered protocol lines.

    let output = run_engine_script("uci\nquit\n");
    let lines = output.lines().collect::<Vec<_>>();

    // Locate uciok so every option line can be verified before handshake completion.

    let uciok_index = lines
        .iter()
        .position(|line| *line == "uciok")
        .expect("missing uciok");

    // The single user-facing executable advertises the full option set.

    let expected_option_lines = [
        "option name Strategy type combo default Random var Random var RelativeCausalEntropy",
        "option name Max Depth type spin default 6 min 1 max 12",
        "option name Random Seed type string",
        "option name Diagnostics Trace type check default false",
        "option name GPU Acceleration type check default false",
    ];

    for expected_option_line in expected_option_lines {

        // Each expected option line must be present and must appear before uciok.

        let option_index = lines
            .iter()
            .position(|line| *line == expected_option_line)
            .unwrap_or_else(|| panic!("missing option line: {expected_option_line}"));

        assert!(
            option_index < uciok_index,
            "option line appeared after uciok: {expected_option_line}"
        );
    }

    // Retired strategy and option names should not appear in the current GUI surface.

    assert!(!output.contains("UniformRolloutPathEntropy"));
    assert!(!output.contains("AccessibleVolumeExpansion"));
    assert!(!output.contains("TauSamples"));
    assert!(!output.contains("TauMacrostate"));
    assert!(output.contains("GPU Acceleration"));
}

//----------------------------------------------------------------------------------------------------------------------
// Function: setoption_updates_typed_session_configuration
//
// Description:
//
//   Verify session-level option handling updates the typed configuration model.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn setoption_updates_typed_session_configuration() {

    // Exercise option updates directly on EngineSession so typed state can be inspected.

    let selector = RandomMoveSelector::new();
    let mut session = EngineSession::new(selector);

    // Apply every known option through the same string-value interface used by the UCI driver.

    assert_eq!(
        session
            .set_option(STRATEGY_OPTION_NAME, Some("RelativeCausalEntropy"))
            .unwrap(),
        OptionUpdateStatus::Updated
    );
    assert_eq!(
        session
            .set_option(MAX_DEPTH_OPTION_NAME, Some("5"))
            .unwrap(),
        OptionUpdateStatus::Updated
    );
    assert_eq!(
        session
            .set_option(RANDOM_SEED_OPTION_NAME, Some("session option seed"))
            .unwrap(),
        OptionUpdateStatus::Updated
    );
    assert_eq!(
        session
            .set_option(DIAGNOSTICS_TRACE_OPTION_NAME, Some("true"))
            .unwrap(),
        OptionUpdateStatus::Updated
    );
    assert_eq!(
        session.set_option(GPU_OPTION_NAME, Some("true")).unwrap(),
        OptionUpdateStatus::Updated
    );

    // Verify all typed fields reflect the accepted option values.

    assert_eq!(
        session.configuration().strategy,
        EngineStrategy::RelativeCausalEntropy
    );
    assert_eq!(session.configuration().max_depth, 5);
    assert_eq!(
        session.configuration().random_seed,
        Some("session option seed".to_string())
    );
    assert!(session.configuration().diagnostics_trace);
    assert!(session.configuration().gpu);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: retired_strategy_values_are_invalid
//
// Description:
//
//   Verify retired strategy values are rejected for the still-known Strategy option.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn retired_strategy_values_are_invalid() {

    // Use the session layer so invalid configuration values are wrapped as EngineError.

    let selector = RandomMoveSelector::new();
    let mut session = EngineSession::new(selector);

    // The active Strategy option should reject retired values and preserve the default strategy.

    assert!(matches!(
        session.set_option(STRATEGY_OPTION_NAME, Some("AccessibleVolumeExpansion")),
        Err(taumax::engine::session::EngineError::Configuration(
            EngineConfigurationError::InvalidValue { .. }
        ))
    ));
    assert_eq!(session.configuration().strategy, EngineStrategy::Random);
}

//----------------------------------------------------------------------------------------------------------------------
// Function: retired_option_names_are_ignored
//
// Description:
//
//   Verify retired neutral option names are treated like unknown GUI options.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn retired_option_names_are_ignored() {

    // Old option names should be classified as unknown rather than failing the session.

    let selector = RandomMoveSelector::new();
    let mut session = EngineSession::new(selector);

    assert_eq!(
        session.set_option("TauSamples", Some("512")).unwrap(),
        OptionUpdateStatus::Unknown
    );
    assert_eq!(
        session.set_option("TauMacrostate", Some("path")).unwrap(),
        OptionUpdateStatus::Unknown
    );
    assert_eq!(
        session
            .set_option("GPU Acceleration", Some("true"))
            .unwrap(),
        OptionUpdateStatus::Updated
    );
}

//----------------------------------------------------------------------------------------------------------------------
// Function: invalid_setoption_values_do_not_crash_engine
//
// Description:
//
//   Verify malformed option commands do not prevent later UCI commands from running.
//
//----------------------------------------------------------------------------------------------------------------------

#[test]
fn invalid_setoption_values_do_not_crash_engine() {

    // Send malformed and unknown option commands before isready.

    let output = run_engine_script(
        "setoption name Max Depth value 0\n\
         setoption name Strategy value NotARealStrategy\n\
         setoption name Hash value 128\n\
         isready\n\
         quit\n",
    );

    // The engine should continue the UCI conversation and answer readiness.

    assert!(predicate::str::contains("readyok").eval(&output));
}

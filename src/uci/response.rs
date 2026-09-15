//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Centralized UCI response formatting.
//
//----------------------------------------------------------------------------------------------------------------------

use crate::engine::configuration::{
    EngineStrategy, DIAGNOSTICS_TRACE_OPTION_NAME, GPU_OPTION_NAME, MAX_DEPTH_DEFAULT,
    MAX_DEPTH_MAXIMUM, MAX_DEPTH_MINIMUM, MAX_DEPTH_OPTION_NAME, RANDOM_SEED_OPTION_NAME,
    STRATEGY_OPTION_NAME,
};
use crate::{ENGINE_AUTHOR, ENGINE_NAME};

//----------------------------------------------------------------------------------------------------------------------
// Function: engine_name
//
// Description:
//
//   Format the UCI engine-name identity response.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn engine_name() -> String {

    // UCI identity lines always start with "id", followed by the field name and value.

    format!("id name {ENGINE_NAME}")
}

//----------------------------------------------------------------------------------------------------------------------
// Function: engine_author
//
// Description:
//
//   Format the UCI engine-author identity response.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn engine_author() -> String {

    // Keep author formatting symmetric with engine_name so the startup handshake is predictable.

    format!("id author {ENGINE_AUTHOR}")
}

//----------------------------------------------------------------------------------------------------------------------
// Function: engine_option_lines
//
// Description:
//
//   Format the UCI option-advertisement lines for Taumax configuration.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn engine_option_lines() -> Vec<String> {

    // Build combo values as repeated "var" entries because UCI combo options have no separate list
    // structure; every allowed value is embedded directly in the option line.

    let strategy_values = EngineStrategy::VALUES
        .into_iter()
        .map(|strategy| format!(" var {strategy}"))
        .collect::<String>();

    // Advertise only options that the session can parse. The text here is the GUI-facing contract.

    vec![
        format!(
            "option name {STRATEGY_OPTION_NAME} type combo default {}{}",
            EngineStrategy::default(),
            strategy_values
        ),
        format!(
            "option name {MAX_DEPTH_OPTION_NAME} type spin default {MAX_DEPTH_DEFAULT} min {MAX_DEPTH_MINIMUM} max {MAX_DEPTH_MAXIMUM}"
        ),
        format!("option name {RANDOM_SEED_OPTION_NAME} type string"),
        format!("option name {DIAGNOSTICS_TRACE_OPTION_NAME} type check default false"),
        format!("option name {GPU_OPTION_NAME} type check default false"),
    ]
}

//----------------------------------------------------------------------------------------------------------------------
// Function: uciok
//
// Description:
//
//   Return the UCI handshake completion response.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn uciok() -> &'static str {

    // "uciok" terminates the startup option advertisement sequence.

    "uciok"
}

//----------------------------------------------------------------------------------------------------------------------
// Function: readyok
//
// Description:
//
//   Return the UCI readiness response.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn readyok() -> &'static str {

    // "readyok" is the synchronization response required by the UCI protocol.

    "readyok"
}

//----------------------------------------------------------------------------------------------------------------------
// Function: bestmove
//
// Description:
//
//   Format a UCI bestmove response for a move text value.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn bestmove(move_text: &str) -> String {

    // The caller supplies either a real UCI move or "0000" for no legal move.

    format!("bestmove {move_text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: bestmove_formats_move
    //
    // Description:
    //
    //   Verify bestmove responses include the selected move text.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn bestmove_formats_move() {

        // Verify the response formatter keeps the move text unchanged after the required keyword.

        assert_eq!(bestmove("e2e4"), "bestmove e2e4");
    }
}

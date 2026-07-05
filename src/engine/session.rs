//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Mutable state for a single UCI engine process.
//
//----------------------------------------------------------------------------------------------------------------------

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cozy_chess::Move;

use crate::board::position::{Position, PositionError};
use crate::engine::configuration::{
    EngineConfiguration, EngineConfigurationError, OptionUpdateStatus,
};
use crate::engine::selector::{MoveSelection, MoveSelector};
use crate::engine::trace::trace_line;
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;
use crate::uci::command::PositionCommand;

//----------------------------------------------------------------------------------------------------------------------
// Struct: EngineSession
//
// Description:
//
//   Stores engine process state, including the current position, selector, and debug flag.
//
//----------------------------------------------------------------------------------------------------------------------

pub struct EngineSession<S>
where
    S: MoveSelector,
{
    position: Position,
    selector: S,
    configuration: EngineConfiguration,
    search_stop_requested: Arc<AtomicBool>,
    debug_enabled: bool,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: EngineSearchResult
//
// Description:
//
//   Stores protocol-ready search output for one go command.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineSearchResult {
    pub best_move_text: String,
    pub trace_lines: Vec<String>,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: EngineError
//
// Description:
//
//   Represents errors surfaced by the engine session layer.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub enum EngineError {
    Position(PositionError),
    Configuration(EngineConfigurationError),
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: EngineSession
//
// Description:
//
//   Provides UCI session state management and move-selection entry points.
//
//----------------------------------------------------------------------------------------------------------------------

impl<S> EngineSession<S>
where
    S: MoveSelector,
{

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create an engine session from a move selector.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(selector: S) -> Self {

        // The session starts from the normal chess opening position. UCI clients can replace it with
        // a later "position" command before asking for a move.

        Self {
            position: Position::startpos(),
            selector,
            configuration: EngineConfiguration::default(),
            search_stop_requested: Arc::new(AtomicBool::new(false)),
            debug_enabled: false,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: configuration
    //
    // Description:
    //
    //   Return the typed engine configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn configuration(&self) -> &EngineConfiguration {

        // Return a shared reference so tests can inspect state without mutating the session.

        &self.configuration
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: new_game
    //
    // Description:
    //
    //   Reset the current position for a new UCI game.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new_game(&mut self) {

        // UCI "ucinewgame" is a hint that previous game history should not influence the next game.

        self.position = Position::startpos();
    }

    //------------------------------------------------------------------------------------------------------------------
    // Mutator: set_debug_enabled
    //
    // Description:
    //
    //   Set whether the session emits diagnostic debug output.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn set_debug_enabled(&mut self, debug_enabled: bool) {

        // Debug mode only affects diagnostics; UCI stdout remains protocol-only.

        self.debug_enabled = debug_enabled;
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: search_stop_flag
    //
    // Description:
    //
    //   Return the shared stop flag used by the UCI input reader.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn search_stop_flag(&self) -> Arc<AtomicBool> {

        // Clone the Arc so the input reader and search control observe the same cancellation flag.

        Arc::clone(&self.search_stop_requested)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: stop_search
    //
    // Description:
    //
    //   Request cooperative cancellation for the current search.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn stop_search(&mut self) {

        // Store with relaxed ordering because the flag carries no data beyond the stop request itself.

        self.search_stop_requested.store(true, Ordering::Relaxed);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: clear_search_stop
    //
    // Description:
    //
    //   Clear any stale stop request before accepting a new game state.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn clear_search_stop(&mut self) {

        // Clear the cooperative stop flag before the next search begins.

        self.search_stop_requested.store(false, Ordering::Relaxed);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Predicate Accessor: is_debug_enabled
    //
    // Description:
    //
    //   Return whether diagnostic debug output is enabled.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn is_debug_enabled(&self) -> bool {

        // The UCI driver uses this to decide whether ignored commands should be echoed to stderr.

        self.debug_enabled
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: set_option
    //
    // Description:
    //
    //   Apply a UCI setoption command to the typed session configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn set_option(
        &mut self,
        option_name: &str,
        value: Option<&str>,
    ) -> Result<OptionUpdateStatus, EngineError> {

        // Delegate validation to EngineConfiguration and lift any configuration error to session scope.

        Ok(self.configuration.set_option(option_name, value)?)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: set_position
    //
    // Description:
    //
    //   Apply a parsed UCI position command atomically to the session.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn set_position(&mut self, command: PositionCommand) -> Result<(), EngineError> {

        // Build the requested position in a temporary value first. If the FEN or a move is bad, the
        // function returns an error and self.position is left unchanged.

        let mut next_position = match command {
            PositionCommand::Startpos { .. } => Position::startpos(),
            PositionCommand::Fen { ref fen, .. } => Position::from_fen(fen)?,
        };

        // UCI represents game history as a base position plus a list of moves. Replaying the moves is
        // what makes castling rights, en-passant state, and side-to-move line up with the GUI.

        for move_text in command.moves() {
            next_position.apply_uci_move(move_text)?;
        }

        // Commit the fully validated position atomically.

        self.position = next_position;

        Ok(())
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: move_selection
    //
    // Description:
    //
    //   Ask the configured selector for a move-selection result in the current position.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn move_selection(&mut self, limits: &SearchLimits, control: &SearchControl) -> MoveSelection {
        self.selector
            .select_move(&self.position, limits, &self.configuration, control)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: best_move
    //
    // Description:
    //
    //   Ask the configured selector for the best move in the current position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn best_move(&mut self, limits: &SearchLimits) -> Option<Move> {

        // The selector is a trait so the engine can swap strategies without changing the UCI driver.

        let control = SearchControl::from_limits(limits);

        // Return only the move component for callers that do not need diagnostics.

        self.move_selection(limits, &control).selected_move
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: search
    //
    // Description:
    //
    //   Select a move and format any enabled trace lines for the UCI driver.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn search(&mut self, limits: &SearchLimits) -> EngineSearchResult {

        // Combine the UCI limits with the shared stop flag so synchronous searches can observe "stop".

        let control = SearchControl::from_limits_and_stop_flag(
            limits,
            Arc::clone(&self.search_stop_requested),
        );

        // Ask the active selector for a move and any strategy-specific score data.

        let move_selection = self.move_selection(limits, &control);
        let best_move_text = match move_selection.selected_move {
            Some(chess_move) => self.position.display_uci_move(chess_move),
            None => "0000".to_string(),
        };

        // Convert diagnostics only when requested; otherwise search returns just the bestmove text.

        let trace_lines = if self.configuration.diagnostics_trace {
            let mut trace_lines = move_selection.diagnostic_lines;

            trace_lines.extend(move_selection.root_move_scores.iter().map(trace_line));

            trace_lines
        } else {
            Vec::new()
        };

        // Package protocol-ready output for the UCI driver.

        EngineSearchResult {
            best_move_text,
            trace_lines,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: best_move_text
    //
    // Description:
    //
    //   Return the selected move as UCI text, or the UCI null move when no move exists.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn best_move_text(&mut self, limits: &SearchLimits) -> String {

        // UCI requires "bestmove 0000" when the side to move has no legal move, such as checkmate or
        // stalemate. Otherwise the move is formatted in UCI coordinate notation.

        self.search(limits).best_move_text
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: From PositionError for EngineError
//
// Description:
//
//   Converts position-layer errors into engine-session errors.
//
//----------------------------------------------------------------------------------------------------------------------

impl From<PositionError> for EngineError {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from
    //
    // Description:
    //
    //   Wrap a PositionError in an EngineError.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn from(error: PositionError) -> Self {

        // Preserve the source error so Display can forward the original message.

        Self::Position(error)
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: From EngineConfigurationError for EngineError
//
// Description:
//
//   Converts configuration-layer errors into engine-session errors.
//
//----------------------------------------------------------------------------------------------------------------------

impl From<EngineConfigurationError> for EngineError {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from
    //
    // Description:
    //
    //   Wrap an EngineConfigurationError in an EngineError.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn from(error: EngineConfigurationError) -> Self {

        // Preserve the source error so option diagnostics keep their detailed context.

        Self::Configuration(error)
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Display for EngineError
//
// Description:
//
//   Formats engine errors as human-readable diagnostic strings.
//
//----------------------------------------------------------------------------------------------------------------------

impl fmt::Display for EngineError {

    //------------------------------------------------------------------------------------------------------------------
    // Method: fmt
    //
    // Description:
    //
    //   Write a readable description of an engine error.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {

        // Forward the wrapped error text without adding protocol-facing decoration.

        match self {
            EngineError::Position(error) => write!(formatter, "{error}"),
            EngineError::Configuration(error) => write!(formatter, "{error}"),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Error for EngineError
//
// Description:
//
//   Marks EngineError as a standard Rust error type.
//
//----------------------------------------------------------------------------------------------------------------------

impl Error for EngineError {}

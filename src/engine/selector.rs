//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Move-selection trait used by engine sessions.
//
//----------------------------------------------------------------------------------------------------------------------

use cozy_chess::Move;

use crate::board::position::Position;
use crate::engine::configuration::EngineConfiguration;
use crate::engine::trace::RootMoveScore;
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Struct: MoveSelection
//
// Description:
//
//   Stores a selected move and any root-move score data produced while selecting it.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct MoveSelection {
    pub selected_move: Option<Move>,
    pub root_move_scores: Vec<RootMoveScore>,
    pub diagnostic_lines: Vec<String>,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: MoveSelection
//
// Description:
//
//   Provides construction behavior for selector results.
//
//----------------------------------------------------------------------------------------------------------------------

impl MoveSelection {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a move-selection result.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(selected_move: Option<Move>, root_move_scores: Vec<RootMoveScore>) -> Self {

        // Start with no standalone diagnostic lines; strategies can attach them when profiling is enabled.

        Self {
            selected_move,
            root_move_scores,
            diagnostic_lines: Vec::new(),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: with_diagnostic_line
    //
    // Description:
    //
    //   Attach a protocol-safe diagnostic line.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_diagnostic_line(mut self, diagnostic_line: String) -> Self {

        // Append the line and return self so selectors can build the result fluently.

        self.diagnostic_lines.push(diagnostic_line);
        self
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_move
    //
    // Description:
    //
    //   Create a result containing only a selected move.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_move(selected_move: Option<Move>) -> Self {

        // Random selection has no scored-root trace data, so it uses the minimal result shape.

        Self::new(selected_move, Vec::new())
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Trait: MoveSelector
//
// Description:
//
//   Defines the move-selection interface used by the UCI engine session.
//
//----------------------------------------------------------------------------------------------------------------------

pub trait MoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Method: select_move
    //
    // Description:
    //
    //   Choose a move for the current position under the requested search limits.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn select_move(
        &mut self,
        position: &Position,
        limits: &SearchLimits,
        configuration: &EngineConfiguration,
        control: &SearchControl,
    ) -> MoveSelection;
}

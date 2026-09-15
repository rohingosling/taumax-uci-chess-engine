//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Terminal-state boundary policy for relative causal entropy.
//
//----------------------------------------------------------------------------------------------------------------------

use cozy_chess::Color;

use crate::board::position::{Position, PositionTerminalState};

// Terminal scores are finite boundaries so they can participate in the same exponential partition
// math as non-terminal freedom scores without producing infinities.

pub const DEFAULT_TERMINAL_WIN_SCORE: f64 = 64.0;
pub const DEFAULT_TERMINAL_LOSS_SCORE: f64 = -64.0;
pub const DEFAULT_TERMINAL_DRAW_SCORE: f64 = 0.0;
//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeTerminalState
//
// Description:
//
//   Classifies terminal states from Taumax's agent-relative perspective.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativeTerminalState {
    Win,
    Loss,
    Draw,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeTerminalState
//
// Description:
//
//   Provides trace formatting for relative terminal-state labels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeTerminalState {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: trace_value
    //
    // Description:
    //
    //   Return the compact UCI-safe label used in Diagnostics Trace output.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn trace_value(self) -> &'static str {

        // Use compact lowercase labels because trace fields are parsed as whitespace-free tokens.

        match self {
            Self::Win => "win",
            Self::Loss => "loss",
            Self::Draw => "draw",
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: TerminalStatePolicy
//
// Description:
//
//   Scores terminal states as future-agency boundary conditions.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalStatePolicy {
    pub win_score: f64,
    pub loss_score: f64,
    pub draw_score: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for TerminalStatePolicy
//
// Description:
//
//   Provides conservative finite terminal boundary scores.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for TerminalStatePolicy {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Return the default terminal-state policy.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {

        // Use symmetric win/loss values and a neutral draw boundary.

        Self {
            win_score: DEFAULT_TERMINAL_WIN_SCORE,
            loss_score: DEFAULT_TERMINAL_LOSS_SCORE,
            draw_score: DEFAULT_TERMINAL_DRAW_SCORE,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: TerminalStatePolicy
//
// Description:
//
//   Classifies and scores terminal boards relative to the Taumax side.
//
//----------------------------------------------------------------------------------------------------------------------

impl TerminalStatePolicy {

    //------------------------------------------------------------------------------------------------------------------
    // Function: classify
    //
    // Description:
    //
    //   Classify a terminal board from Taumax's perspective.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn classify(
        &self,
        position: &Position,
        taumax_side: Color,
    ) -> Option<RelativeTerminalState> {

        // A non-terminal board returns None immediately; only ended games receive boundary labels.

        match position.terminal_state()? {
            PositionTerminalState::Checkmate => {

                // In checkmate, the side to move is the side that has been mated.

                if position.board().side_to_move() == taumax_side {
                    Some(RelativeTerminalState::Loss)
                } else {
                    Some(RelativeTerminalState::Win)
                }
            }
            PositionTerminalState::Draw => Some(RelativeTerminalState::Draw),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: score
    //
    // Description:
    //
    //   Return the scalar boundary value for a relative terminal state.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn score(&self, terminal_state: RelativeTerminalState) -> f64 {

        // Map the classified terminal state onto the configured scalar boundary value.

        match terminal_state {
            RelativeTerminalState::Win => self.win_score,
            RelativeTerminalState::Loss => self.loss_score,
            RelativeTerminalState::Draw => self.draw_score,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: score_position
    //
    // Description:
    //
    //   Classify and score a terminal board, if the board is terminal.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn score_position(
        &self,
        position: &Position,
        taumax_side: Color,
    ) -> Option<(RelativeTerminalState, f64)> {

        // Compute classification first, then return it alongside the matching score for trace output.

        let terminal_state = self.classify(position, taumax_side)?;

        Some((terminal_state, self.score(terminal_state)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: checkmating_opponent_is_terminal_win
    //
    // Description:
    //
    //   Verify a checkmated opponent is scored as a Taumax win boundary.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn checkmating_opponent_is_terminal_win() {

        // Black is checkmated in this FEN, so White receives the win boundary.

        let position = Position::from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1").unwrap();
        let policy = TerminalStatePolicy::default();

        assert_eq!(
            policy.classify(&position, Color::White),
            Some(RelativeTerminalState::Win)
        );
        assert_eq!(
            policy.score_position(&position, Color::White),
            Some((RelativeTerminalState::Win, DEFAULT_TERMINAL_WIN_SCORE))
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: being_checkmated_is_terminal_loss
    //
    // Description:
    //
    //   Verify a checkmated Taumax side is scored as a loss boundary.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn being_checkmated_is_terminal_loss() {

        // The same board is a loss when evaluated from Black's Taumax perspective.

        let position = Position::from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1").unwrap();
        let policy = TerminalStatePolicy::default();

        assert_eq!(
            policy.classify(&position, Color::Black),
            Some(RelativeTerminalState::Loss)
        );
        assert_eq!(
            policy.score_position(&position, Color::Black),
            Some((RelativeTerminalState::Loss, DEFAULT_TERMINAL_LOSS_SCORE))
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: stalemate_is_terminal_draw
    //
    // Description:
    //
    //   Verify stalemate receives its own draw boundary.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn stalemate_is_terminal_draw() {

        // Black has no legal move but is not in check, so the boundary is a draw for either side.

        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let policy = TerminalStatePolicy::default();

        assert_eq!(
            policy.classify(&position, Color::White),
            Some(RelativeTerminalState::Draw)
        );
        assert_eq!(
            policy.score_position(&position, Color::White),
            Some((RelativeTerminalState::Draw, DEFAULT_TERMINAL_DRAW_SCORE))
        );
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Project-specific position adapter around cozy-chess.
//
//----------------------------------------------------------------------------------------------------------------------

use std::error::Error;
use std::fmt;

use cozy_chess::util::{display_uci_move as display_cozy_uci_move, parse_uci_move};
use cozy_chess::{Board, GameStatus, Move};

//----------------------------------------------------------------------------------------------------------------------
// Struct: Position
//
// Description:
//
//   Wraps the cozy-chess board so the rest of the project depends on a local position type.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Position {

    // cozy-chess owns the rules of chess: legal move generation, FEN parsing, side-to-move tracking,
    // castling rights, and en-passant state.

    board: Board,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: PositionTerminalState
//
// Description:
//
//   Classifies terminal game states from the current side-to-move perspective.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionTerminalState {
    Checkmate,
    Draw,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: PositionError
//
// Description:
//
//   Represents position setup and move-application failures in project-specific terms.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PositionError {

    // FEN describes a full chess position in one text string.

    InvalidFen { text: String, message: String },

    // Move text did not have valid UCI coordinate notation for the current board.

    InvalidMoveText { text: String, message: String },

    // Move text was syntactically valid but not legal in the current chess position.

    IllegalMove { text: String, message: String },
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Position
//
// Description:
//
//   Provides constructors, move application, move listing, and display helpers for a chess position.
//
//----------------------------------------------------------------------------------------------------------------------

impl Position {

    //------------------------------------------------------------------------------------------------------------------
    // Function: startpos
    //
    // Description:
    //
    //   Create a position initialized to the standard chess starting position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn startpos() -> Self {
        Self {
            board: Board::startpos(),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_fen
    //
    // Description:
    //
    //   Parse a FEN string into a position and translate parse failures to PositionError.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {

        // Rust's parse method delegates to cozy-chess's FromStr implementation for Board.

        let board = fen
            .parse::<Board>()
            .map_err(|error| PositionError::InvalidFen {
                text: fen.to_string(),
                message: error.to_string(),
            })?;

        // Return the parsed board inside the local adapter type used by the rest of the engine.

        Ok(Self { board })
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: apply_uci_move
    //
    // Description:
    //
    //   Parse and apply one UCI move to the current position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn apply_uci_move(&mut self, text: &str) -> Result<(), PositionError> {

        // UCI moves are written as source square plus destination square, with an optional promotion
        // piece: "e2e4", "g1f3", or "a7a8q".

        let chess_move =
            parse_uci_move(&self.board, text).map_err(|error| PositionError::InvalidMoveText {
                text: text.to_string(),
                message: error.to_string(),
            })?;

        // try_play checks legality before mutating the board, so illegal moves become errors instead
        // of corrupting the position.

        self.board
            .try_play(chess_move)
            .map_err(|error| PositionError::IllegalMove {
                text: text.to_string(),
                message: error.to_string(),
            })
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: apply_move
    //
    // Description:
    //
    //   Apply an already parsed chess move to the current position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn apply_move(&mut self, chess_move: Move) -> Result<(), PositionError> {

        // Format the move before mutation so any legality error can report the original UCI text.

        let move_text = self.display_uci_move(chess_move);

        // Apply the generated move through cozy-chess so legality and state updates stay rule-correct.

        self.board
            .try_play(chess_move)
            .map_err(|error| PositionError::IllegalMove {
                text: move_text,
                message: error.to_string(),
            })
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: legal_moves
    //
    // Description:
    //
    //   Collect all legal moves available in the current position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut legal_moves = Vec::new();

        // cozy-chess groups generated moves by piece. The callback appends each group into one flat
        // vector because this simple engine only needs a list it can choose from.

        self.board.generate_moves(|piece_moves| {
            legal_moves.extend(piece_moves);

            // Returning false tells cozy-chess to keep generating moves. A true return value would be
            // useful for early-exit searches that only need to know whether at least one move exists.

            false
        });

        legal_moves
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: display_uci_move
    //
    // Description:
    //
    //   Format a cozy-chess move as UCI text in the context of the current board.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn display_uci_move(&self, chess_move: Move) -> String {
        format!("{}", display_cozy_uci_move(&self.board, chess_move))
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: terminal_state
    //
    // Description:
    //
    //   Return the board's terminal status, if the game has ended.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn terminal_state(&self) -> Option<PositionTerminalState> {

        // Translate cozy-chess terminal status into the smaller terminal-state vocabulary used here.

        match self.board.status() {
            GameStatus::Ongoing => None,
            GameStatus::Won => Some(PositionTerminalState::Checkmate),
            GameStatus::Drawn => Some(PositionTerminalState::Draw),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: board
    //
    // Description:
    //
    //   Return the underlying cozy-chess board.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn board(&self) -> &Board {
        &self.board
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for Position
//
// Description:
//
//   Provides the standard chess starting position as the default position value.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for Position {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Return a position initialized to the standard chess starting position.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {
        Self::startpos()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Display for PositionError
//
// Description:
//
//   Formats position errors as human-readable diagnostic strings.
//
//----------------------------------------------------------------------------------------------------------------------

impl fmt::Display for PositionError {

    //------------------------------------------------------------------------------------------------------------------
    // Method: fmt
    //
    // Description:
    //
    //   Write a readable description of a position error.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {

        // Select the error variant and write a diagnostic that includes the rejected input.

        match self {

            // Include both the failed FEN and cozy-chess's explanation so bad setup commands are easy
            // to diagnose from stderr.

            PositionError::InvalidFen { text, message } => {
                write!(formatter, "invalid FEN '{text}': {message}")
            }

            // InvalidMoveText means the move string could not be parsed in this board context.

            PositionError::InvalidMoveText { text, message } => {
                write!(formatter, "invalid UCI move '{text}': {message}")
            }

            // IllegalMove means the notation parsed, but chess rules rejected the move.

            PositionError::IllegalMove { text, message } => {
                write!(formatter, "illegal UCI move '{text}': {message}")
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Error for PositionError
//
// Description:
//
//   Marks PositionError as a standard Rust error type.
//
//----------------------------------------------------------------------------------------------------------------------

impl Error for PositionError {}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: startpos_has_legal_moves
    //
    // Description:
    //
    //   Verify the standard chess starting position exposes twenty legal moves.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn startpos_has_legal_moves() {

        // The standard opening position has twenty legal moves for White.

        let position = Position::startpos();

        assert_eq!(position.legal_moves().len(), 20);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: legal_uci_move_can_be_applied
    //
    // Description:
    //
    //   Verify a legal UCI move mutates the position and changes the side to move.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn legal_uci_move_can_be_applied() {

        // Apply the common opening move e2e4 from the starting position.

        let mut position = Position::startpos();

        position.apply_uci_move("e2e4").unwrap();

        // After White moves, Black must be the side to move.

        assert_eq!(position.board().side_to_move(), cozy_chess::Color::Black);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: legal_generated_move_can_be_applied
    //
    // Description:
    //
    //   Verify generated moves can be applied without round-tripping through UCI text.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn legal_generated_move_can_be_applied() {

        // Take a move generated by cozy-chess so it is guaranteed legal in the current position.

        let mut position = Position::startpos();
        let chess_move = position.legal_moves()[0];

        position.apply_move(chess_move).unwrap();

        // Applying any legal White move should pass the turn to Black.

        assert_eq!(position.board().side_to_move(), cozy_chess::Color::Black);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: illegal_uci_move_returns_error
    //
    // Description:
    //
    //   Verify an illegal UCI move is rejected with an error.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn illegal_uci_move_returns_error() {

        // A king move from e1 to e8 is not legal in the starting position.

        let mut position = Position::startpos();

        assert!(position.apply_uci_move("e1e8").is_err());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: checkmate_position_reports_terminal_checkmate
    //
    // Description:
    //
    //   Verify the adapter distinguishes a checkmated side from ordinary no-move mobility collapse.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn checkmate_position_reports_terminal_checkmate() {

        // Black to move is checkmated by the White king and queen.

        let position = Position::from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1").unwrap();

        // The adapter should report checkmate and expose no legal moves.

        assert_eq!(
            position.terminal_state(),
            Some(PositionTerminalState::Checkmate)
        );
        assert!(position.legal_moves().is_empty());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: stalemate_position_reports_terminal_draw
    //
    // Description:
    //
    //   Verify the adapter distinguishes stalemate from checkmate.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn stalemate_position_reports_terminal_draw() {

        // Black to move has no legal moves but is not in check.

        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();

        // The adapter should report draw rather than checkmate.

        assert_eq!(position.terminal_state(), Some(PositionTerminalState::Draw));
        assert!(position.legal_moves().is_empty());
    }
}

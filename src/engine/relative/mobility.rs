//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Legal-move mobility primitives for the relative causal-entropy strategy.
//
//----------------------------------------------------------------------------------------------------------------------

use std::collections::{BTreeMap, BTreeSet};

use cozy_chess::{Color, Move, Piece, Square};

use crate::board::position::Position;

//----------------------------------------------------------------------------------------------------------------------
// Struct: PieceMobility
//
// Description:
//
//   Counts the legal destination squares available to one physical piece.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PieceMobility {
    pub square: Square,
    pub piece: Piece,
    pub legal_destination_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: PieceMobilityProfile
//
// Description:
//
//   Aggregates legal destination counts per physical piece for the natural side to move.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PieceMobilityProfile {
    piece_mobilities: Vec<PieceMobility>,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: PieceMobilityProfile
//
// Description:
//
//   Builds and queries per-piece legal mobility summaries.
//
//----------------------------------------------------------------------------------------------------------------------

impl PieceMobilityProfile {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a per-piece mobility profile with deterministic square ordering.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(mut piece_mobilities: Vec<PieceMobility>) -> Self {

        // Sort by square so downstream diagnostics and tests see deterministic ordering.

        piece_mobilities.sort_by_key(|piece_mobility| piece_mobility.square);

        Self { piece_mobilities }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Measure per-piece legal destinations from the current legal side-to-move position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(position: &Position) -> Self {

        // Generate legal moves once, then reuse that list to build per-piece destination counts.

        let legal_moves = position.legal_moves();

        Self::from_legal_moves(position, &legal_moves)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: piece_mobilities
    //
    // Description:
    //
    //   Return the sorted per-piece mobility entries.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn piece_mobilities(&self) -> &[PieceMobility] {

        // Expose a slice so callers can inspect the sorted profile without mutating it.

        &self.piece_mobilities
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: piece_mobility_for
    //
    // Description:
    //
    //   Return the mobility entry for a source square, when that piece has legal destinations.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn piece_mobility_for(&self, square: Square) -> Option<&PieceMobility> {

        // Only pieces with at least one legal destination appear in the profile.

        self.piece_mobilities
            .iter()
            .find(|piece_mobility| piece_mobility.square == square)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: legal_destination_count_for
    //
    // Description:
    //
    //   Return the number of legal destination squares for a source square.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn legal_destination_count_for(&self, square: Square) -> usize {

        // Missing entries mean the piece has zero legal destinations from that square.

        self.piece_mobility_for(square)
            .map(|piece_mobility| piece_mobility.legal_destination_count)
            .unwrap_or_default()
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: total_legal_destination_count
    //
    // Description:
    //
    //   Return the summed legal destination count across pieces with at least one legal destination.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn total_legal_destination_count(&self) -> usize {

        // Compute the total by summing every per-piece destination count in the profile.

        self.piece_mobilities
            .iter()
            .map(|piece_mobility| piece_mobility.legal_destination_count)
            .sum()
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_legal_moves
    //
    // Description:
    //
    //   Aggregate unique legal destination squares by source square from generated legal moves.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn from_legal_moves(position: &Position, legal_moves: &[Move]) -> Self {

        // Use ordered maps and sets so aggregation is deterministic regardless of move generation order.

        let side_to_move = position.board().side_to_move();
        let mut legal_destinations_by_square: BTreeMap<Square, (Piece, BTreeSet<Square>)> =
            BTreeMap::new();

        for chess_move in legal_moves {

            // Each legal move source must contain the moving side's piece.

            let moving_piece = position
                .board()
                .piece_on(chess_move.from)
                .expect("legal move source square must contain a piece");

            debug_assert_eq!(
                position.board().color_on(chess_move.from),
                Some(side_to_move)
            );

            // Group destination squares by the physical source square of the moving piece.

            let (_, legal_destinations) = legal_destinations_by_square
                .entry(chess_move.from)
                .or_insert_with(|| (moving_piece, BTreeSet::new()));

            legal_destinations.insert(chess_move.to);
        }

        // Convert each grouped destination set into a compact count for scoring.

        let piece_mobilities = legal_destinations_by_square
            .into_iter()
            .map(|(square, (piece, legal_destinations))| PieceMobility {
                square,
                piece,
                legal_destination_count: legal_destinations.len(),
            })
            .collect();

        Self { piece_mobilities }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: SideMobility
//
// Description:
//
//   Summarizes legal mobility for the natural side to move in one position.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SideMobility {
    pub side_to_move: Color,
    pub legal_move_count: usize,
    pub piece_mobility_profile: PieceMobilityProfile,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: SideMobility
//
// Description:
//
//   Measures side-to-move mobility through legal move generation.
//
//----------------------------------------------------------------------------------------------------------------------

impl SideMobility {

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Measure legal mobility for the current side to move.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(position: &Position) -> Self {

        // Measure global legal moves and the per-piece profile from the same generated move list.

        let legal_moves = position.legal_moves();
        let piece_mobility_profile = PieceMobilityProfile::from_legal_moves(position, &legal_moves);

        Self {
            side_to_move: position.board().side_to_move(),
            legal_move_count: legal_moves.len(),
            piece_mobility_profile,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeMobility
//
// Description:
//
//   Compares Taumax legal mobility against weighted opponent legal mobility.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct RelativeMobility {
    pub taumax_mobility: SideMobility,
    pub opponent_mobility: SideMobility,
    pub opponent_weight: f64,
    pub mobility_difference: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeMobility
//
// Description:
//
//   Builds relative mobility summaries from naturally alternating positions.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeMobility {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a relative mobility summary from already measured side mobilities.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(
        taumax_mobility: SideMobility,
        opponent_mobility: SideMobility,
        opponent_weight: f64,
    ) -> Self {

        // Compute relative mobility as Taumax legal moves minus weighted opponent legal moves.

        let mobility_difference = taumax_mobility.legal_move_count as f64
            - opponent_weight * opponent_mobility.legal_move_count as f64;

        // Store both raw mobility objects so callers can inspect the terms behind the difference.

        Self {
            taumax_mobility,
            opponent_mobility,
            opponent_weight,
            mobility_difference,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Measure Taumax and opponent mobility from positions where each side naturally has the move.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(
        taumax_position: &Position,
        opponent_position: &Position,
        opponent_weight: f64,
    ) -> Self {

        // Measure each side in the position where that side is naturally to move.

        Self::new(
            SideMobility::measure(taumax_position),
            SideMobility::measure(opponent_position),
            opponent_weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: starting_position_side_mobility_counts_are_stable
    //
    // Description:
    //
    //   Verify startpos legal move and per-piece destination counts.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn starting_position_side_mobility_counts_are_stable() {

        // Startpos has twenty legal moves distributed across ten mobile pieces.

        let position = Position::startpos();

        let mobility = SideMobility::measure(&position);

        assert_eq!(mobility.side_to_move, Color::White);
        assert_eq!(mobility.legal_move_count, 20);
        assert_eq!(mobility.piece_mobility_profile.piece_mobilities().len(), 10);
        assert_eq!(
            mobility
                .piece_mobility_profile
                .total_legal_destination_count(),
            20
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: piece_mobility_profile_aggregates_by_source_square
    //
    // Description:
    //
    //   Verify generated legal moves are grouped by the physical piece that can make them.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn piece_mobility_profile_aggregates_by_source_square() {

        // Use startpos because both knights and pawns have simple known destination counts.

        let position = Position::startpos();

        // Query the b1 knight directly to prove source-square aggregation works.

        let mobility = SideMobility::measure(&position);
        let knight_mobility = mobility
            .piece_mobility_profile
            .piece_mobility_for(Square::B1)
            .expect("b1 knight should have legal moves");

        assert_eq!(knight_mobility.piece, Piece::Knight);
        assert_eq!(knight_mobility.legal_destination_count, 2);
        assert_eq!(
            mobility
                .piece_mobility_profile
                .legal_destination_count_for(Square::A2),
            2
        );
        assert_eq!(
            mobility
                .piece_mobility_profile
                .legal_destination_count_for(Square::A1),
            0
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: pinned_piece_has_no_pseudo_legal_mobility
    //
    // Description:
    //
    //   Verify pinned pieces are measured through legal move generation rather than attack maps.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn pinned_piece_has_no_pseudo_legal_mobility() {

        // The e2 knight is pinned against the white king, so pseudo-legal jumps must not count.

        let position = Position::from_fen("k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1").unwrap();

        let mobility = SideMobility::measure(&position);

        assert_eq!(position.board().piece_on(Square::E2), Some(Piece::Knight));
        assert_eq!(
            mobility
                .piece_mobility_profile
                .legal_destination_count_for(Square::E2),
            0
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_mobility_compares_natural_side_to_move_positions
    //
    // Description:
    //
    //   Verify relative mobility uses naturally alternating positions for Taumax and opponent measurements.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn relative_mobility_compares_natural_side_to_move_positions() {

        // Build opponent and Taumax positions on alternating turns from the same opening line.

        let mut opponent_position = Position::startpos();
        opponent_position.apply_uci_move("e2e4").unwrap();

        let mut taumax_position = opponent_position.clone();
        taumax_position.apply_uci_move("e7e5").unwrap();

        // Compute the expected subtraction explicitly from the measured raw move counts.

        let relative_mobility =
            RelativeMobility::measure(&taumax_position, &opponent_position, 0.5);
        let expected_mobility_difference = relative_mobility.taumax_mobility.legal_move_count
            as f64
            - 0.5 * relative_mobility.opponent_mobility.legal_move_count as f64;

        assert_eq!(relative_mobility.taumax_mobility.side_to_move, Color::White);
        assert_eq!(
            relative_mobility.opponent_mobility.side_to_move,
            Color::Black
        );
        assert!(
            relative_mobility.taumax_mobility.legal_move_count
                > relative_mobility.opponent_mobility.legal_move_count
        );
        assert!(
            (relative_mobility.mobility_difference - expected_mobility_difference).abs()
                < f64::EPSILON
        );
    }
}

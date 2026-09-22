//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Immediate freedom-potential scoring for the relative causal-entropy strategy.
//
//----------------------------------------------------------------------------------------------------------------------

use crate::board::position::Position;
use crate::engine::relative::mobility::SideMobility;

pub const DEFAULT_PIECE_MOBILITY_WEIGHT: f64 = 0.25;
pub const DEFAULT_OPPONENT_FREEDOM_WEIGHT: f64 = 1.0;

//----------------------------------------------------------------------------------------------------------------------
// Struct: FreedomPotentialWeights
//
// Description:
//
//   Contains the internal weights used by the immediate freedom potential.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreedomPotentialWeights {
    pub piece_mobility_weight: f64,
    pub opponent_freedom_weight: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for FreedomPotentialWeights
//
// Description:
//
//   Provides the initial unexposed relative-potential weights.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for FreedomPotentialWeights {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Return the default internal weights for the first relative implementation.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {
        Self {
            piece_mobility_weight: DEFAULT_PIECE_MOBILITY_WEIGHT,
            opponent_freedom_weight: DEFAULT_OPPONENT_FREEDOM_WEIGHT,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: SideFreedom
//
// Description:
//
//   Scores the immediate freedom available to one naturally side-to-move player.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct SideFreedom {
    pub side_mobility: SideMobility,
    pub legal_move_term: f64,
    pub piece_mobility_term: f64,
    pub value: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: SideFreedom
//
// Description:
//
//   Computes logarithmic freedom terms from legal side mobility.
//
//----------------------------------------------------------------------------------------------------------------------

impl SideFreedom {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_mobility
    //
    // Description:
    //
    //   Score immediate side freedom from already measured legal mobility.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_mobility(side_mobility: SideMobility, piece_mobility_weight: f64) -> Self {

        // Compute the legal-move term as ln(1 + moves), keeping zero mobility finite.

        let legal_move_term = logarithmic_count(side_mobility.legal_move_count);

        // Compute piece mobility by summing ln(1 + destinations) for each piece with legal moves.

        let piece_mobility_term = side_mobility
            .piece_mobility_profile
            .piece_mobilities()
            .iter()
            .map(|piece_mobility| logarithmic_count(piece_mobility.legal_destination_count))
            .sum::<f64>();

        // Compute side freedom as global legal moves plus the weighted per-piece mobility sum.

        let value = legal_move_term + piece_mobility_weight * piece_mobility_term;

        // Store the source mobility and every term so diagnostics can show how the score was built.

        Self {
            side_mobility,
            legal_move_term,
            piece_mobility_term,
            value,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Measure and score immediate side freedom from the natural side-to-move position.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(position: &Position, piece_mobility_weight: f64) -> Self {
        Self::from_mobility(SideMobility::measure(position), piece_mobility_weight)
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeFreedomPotential
//
// Description:
//
//   Scores Taumax freedom relative to weighted opponent freedom.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct RelativeFreedomPotential {
    pub taumax_freedom: SideFreedom,
    pub opponent_freedom: SideFreedom,
    pub weights: FreedomPotentialWeights,
    pub value: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeFreedomPotential
//
// Description:
//
//   Builds relative freedom-potential scores from legal mobility measurements.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeFreedomPotential {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_mobilities
    //
    // Description:
    //
    //   Score relative freedom from naturally measured Taumax and opponent mobilities.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_mobilities(
        taumax_mobility: SideMobility,
        opponent_mobility: SideMobility,
        weights: FreedomPotentialWeights,
    ) -> Self {

        // Compute Taumax freedom from its legal mobility profile.

        let taumax_freedom =
            SideFreedom::from_mobility(taumax_mobility, weights.piece_mobility_weight);

        // Compute opponent freedom with the same piece-mobility scale for a fair comparison.

        let opponent_freedom =
            SideFreedom::from_mobility(opponent_mobility, weights.piece_mobility_weight);

        // Compute relative freedom as Taumax freedom minus weighted opponent freedom.

        let value = taumax_freedom.value - weights.opponent_freedom_weight * opponent_freedom.value;

        // Store both component scores and the weights used to combine them.

        Self {
            taumax_freedom,
            opponent_freedom,
            weights,
            value,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Measure and score relative freedom from positions where each side naturally has the move.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(
        taumax_position: &Position,
        opponent_position: &Position,
        weights: FreedomPotentialWeights,
    ) -> Self {

        // Measure both natural side-to-move positions before applying the relative freedom formula.

        Self::from_mobilities(
            SideMobility::measure(taumax_position),
            SideMobility::measure(opponent_position),
            weights,
        )
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: logarithmic_count
//
// Description:
//
//   Return log(1 + count) for non-negative mobility counts.
//
//----------------------------------------------------------------------------------------------------------------------

fn logarithmic_count(count: usize) -> f64 {
    (1.0 + count as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::{Color, Piece, Square};

    use crate::engine::relative::mobility::{PieceMobility, PieceMobilityProfile};

    const FLOAT_TOLERANCE: f64 = 1.0e-12;

    //------------------------------------------------------------------------------------------------------------------
    // Function: assert_close
    //
    // Description:
    //
    //   Verify two floating-point values are equal within the local test tolerance.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn assert_close(left_value: f64, right_value: f64) {
        assert!(
            (left_value - right_value).abs() < FLOAT_TOLERANCE,
            "left={left_value}, right={right_value}"
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: side_mobility_with_piece_counts
    //
    // Description:
    //
    //   Build a synthetic side-mobility value for formula-focused tests.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn side_mobility_with_piece_counts(
        side_to_move: Color,
        legal_move_count: usize,
        piece_counts: Vec<(Square, Piece, usize)>,
    ) -> SideMobility {

        // Convert compact tuple inputs into synthetic piece-mobility records for formula tests.

        let piece_mobilities = piece_counts
            .into_iter()
            .map(|(square, piece, legal_destination_count)| PieceMobility {
                square,
                piece,
                legal_destination_count,
            })
            .collect();

        // Assemble the side-mobility value with the requested legal-move count and piece profile.

        SideMobility {
            side_to_move,
            legal_move_count,
            piece_mobility_profile: PieceMobilityProfile::new(piece_mobilities),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: startpos_side_freedom_matches_formula
    //
    // Description:
    //
    //   Verify the side-freedom score uses the configured logarithmic count formula.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn startpos_side_freedom_matches_formula() {

        // Create the standard opening position whose mobility counts are known and stable.

        let position = Position::startpos();

        // Compute the expected formula: twenty legal moves give ln(21), and ten pieces with two
        // destinations each give ten times ln(3).

        let side_freedom = SideFreedom::measure(&position, DEFAULT_PIECE_MOBILITY_WEIGHT);
        let expected_piece_mobility_term = 10.0 * (3.0_f64).ln();
        let expected_value =
            (21.0_f64).ln() + DEFAULT_PIECE_MOBILITY_WEIGHT * expected_piece_mobility_term;

        // Compare the legal-move term against ln(1 + 20).

        assert_close(side_freedom.legal_move_term, (21.0_f64).ln());

        // Compare the summed piece-mobility term against ten copies of ln(1 + 2).

        assert_close(
            side_freedom.piece_mobility_term,
            expected_piece_mobility_term,
        );

        // Compare the final freedom score against legal moves plus weighted piece mobility.

        assert_close(side_freedom.value, expected_value);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: piece_identity_does_not_change_side_freedom
    //
    // Description:
    //
    //   Verify the side-freedom formula does not use material values or piece identity weights.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn piece_identity_does_not_change_side_freedom() {

        // Build a synthetic mobility profile using a knight and pawn.

        let knight_mobility = side_mobility_with_piece_counts(
            Color::White,
            4,
            vec![(Square::A1, Piece::Knight, 3), (Square::B1, Piece::Pawn, 1)],
        );

        // Build the same mobility counts with a queen replacing the knight.

        let queen_mobility = side_mobility_with_piece_counts(
            Color::White,
            4,
            vec![(Square::A1, Piece::Queen, 3), (Square::B1, Piece::Pawn, 1)],
        );

        // Compute freedom from the knight profile using only legal-destination counts.

        let knight_freedom =
            SideFreedom::from_mobility(knight_mobility, DEFAULT_PIECE_MOBILITY_WEIGHT);

        // Compute freedom from the queen profile with the same mobility counts.

        let queen_freedom =
            SideFreedom::from_mobility(queen_mobility, DEFAULT_PIECE_MOBILITY_WEIGHT);

        assert_close(knight_freedom.value, queen_freedom.value);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_potential_increases_with_taumax_mobility
    //
    // Description:
    //
    //   Verify additional Taumax freedom raises the relative score when opponent freedom is unchanged.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn relative_potential_increases_with_taumax_mobility() {

        // Build fixed opponent mobility so only the Taumax side changes between cases.

        let opponent_mobility =
            side_mobility_with_piece_counts(Color::Black, 5, vec![(Square::H8, Piece::King, 5)]);

        // Build the lower Taumax mobility case.

        let constrained_taumax_mobility =
            side_mobility_with_piece_counts(Color::White, 4, vec![(Square::G1, Piece::King, 4)]);

        // Build the higher Taumax mobility case by adding legal moves and one mobile piece.

        let freer_taumax_mobility = side_mobility_with_piece_counts(
            Color::White,
            8,
            vec![(Square::G1, Piece::King, 4), (Square::B1, Piece::Knight, 2)],
        );

        // Compute the relative score for the lower Taumax mobility case.

        let constrained_potential = RelativeFreedomPotential::from_mobilities(
            constrained_taumax_mobility,
            opponent_mobility.clone(),
            FreedomPotentialWeights::default(),
        );

        // Compute the relative score for the higher Taumax mobility case.

        let freer_potential = RelativeFreedomPotential::from_mobilities(
            freer_taumax_mobility,
            opponent_mobility,
            FreedomPotentialWeights::default(),
        );

        // Verify increasing Taumax freedom raises the relative score when opponent freedom is fixed.

        assert!(freer_potential.value > constrained_potential.value);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_potential_decreases_with_opponent_mobility
    //
    // Description:
    //
    //   Verify additional opponent freedom lowers the relative score when Taumax freedom is unchanged.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn relative_potential_decreases_with_opponent_mobility() {

        // Build fixed Taumax mobility so only the opponent side changes between cases.

        let taumax_mobility =
            side_mobility_with_piece_counts(Color::White, 8, vec![(Square::G1, Piece::King, 4)]);

        // Build the lower opponent mobility case.

        let constrained_opponent_mobility =
            side_mobility_with_piece_counts(Color::Black, 4, vec![(Square::H8, Piece::King, 4)]);

        // Build the higher opponent mobility case by adding legal moves and one mobile piece.

        let freer_opponent_mobility = side_mobility_with_piece_counts(
            Color::Black,
            10,
            vec![(Square::H8, Piece::King, 4), (Square::B8, Piece::Knight, 3)],
        );

        // Compute the relative score for the lower opponent mobility case.

        let constrained_potential = RelativeFreedomPotential::from_mobilities(
            taumax_mobility.clone(),
            constrained_opponent_mobility,
            FreedomPotentialWeights::default(),
        );

        // Compute the relative score for the higher opponent mobility case.

        let freer_opponent_potential = RelativeFreedomPotential::from_mobilities(
            taumax_mobility,
            freer_opponent_mobility,
            FreedomPotentialWeights::default(),
        );

        // Verify increasing opponent freedom lowers the relative score through the subtraction term.

        assert!(freer_opponent_potential.value < constrained_potential.value);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_potential_measures_natural_side_to_move_positions
    //
    // Description:
    //
    //   Verify the measured relative potential keeps each side on naturally alternating turns.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn relative_potential_measures_natural_side_to_move_positions() {

        // Advance from the starting position once so the opponent position has Black to move.

        let mut opponent_position = Position::startpos();
        opponent_position.apply_uci_move("e2e4").unwrap();

        // Clone the opponent position and play Black's reply so the Taumax position has White to move.

        let mut taumax_position = opponent_position.clone();
        taumax_position.apply_uci_move("e7e5").unwrap();

        // Compute the measured relative potential and the explicit subtraction expected from the formula.

        let relative_potential = RelativeFreedomPotential::measure(
            &taumax_position,
            &opponent_position,
            FreedomPotentialWeights::default(),
        );
        let expected_value =
            relative_potential.taumax_freedom.value - relative_potential.opponent_freedom.value;

        // Verify the Taumax component was measured from the position where White is to move.

        assert_eq!(
            relative_potential.taumax_freedom.side_mobility.side_to_move,
            Color::White
        );

        // Verify the opponent component was measured from the position where Black is to move.

        assert_eq!(
            relative_potential
                .opponent_freedom
                .side_mobility
                .side_to_move,
            Color::Black
        );

        // Compare the measured value with Taumax freedom minus opponent freedom.

        assert_close(relative_potential.value, expected_value);
    }
}

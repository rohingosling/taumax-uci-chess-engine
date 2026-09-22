//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Random legal move selector.
//
//----------------------------------------------------------------------------------------------------------------------

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::board::position::Position;
use crate::engine::configuration::EngineConfiguration;
use crate::engine::seed::optional_seeded_random_number_generator;
use crate::engine::selector::{MoveSelection, MoveSelector};
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Struct: RandomMoveSelector
//
// Description:
//
//   Stores the random number generator used by the random legal move selector.
//
//----------------------------------------------------------------------------------------------------------------------

pub struct RandomMoveSelector {
    random_number_generator: ThreadRng,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RandomMoveSelector
//
// Description:
//
//   Provides construction behavior for the random legal move selector.
//
//----------------------------------------------------------------------------------------------------------------------

impl RandomMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a random move selector with a thread-local random number generator.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new() -> Self {
        Self {
            random_number_generator: rand::thread_rng(),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for RandomMoveSelector
//
// Description:
//
//   Uses the normal constructor as the selector default.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for RandomMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Create the default random move selector.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {
        Self::new()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: MoveSelector for RandomMoveSelector
//
// Description:
//
//   Selects one legal move at random from the current position.
//
//----------------------------------------------------------------------------------------------------------------------

impl MoveSelector for RandomMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Method: select_move
    //
    // Description:
    //
    //   Return a randomly selected legal move, or None when no legal moves exist.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn select_move(
        &mut self,
        position: &Position,
        limits: &SearchLimits,
        configuration: &EngineConfiguration,
        _control: &SearchControl,
    ) -> MoveSelection {

        // Build the candidate list from legal moves and the optional searchmoves root filter.

        let mut legal_moves = position
            .legal_moves()
            .into_iter()
            .filter(|chess_move| {
                let move_text = position.display_uci_move(*chess_move);

                limits.allows_root_move(&move_text)
            })
            .collect::<Vec<_>>();

        // Sort before random choice so a fixed seed sees the same candidate ordering on every run.

        legal_moves.sort_by(|left_move, right_move| {
            let left_move_text = position.display_uci_move(*left_move);
            let right_move_text = position.display_uci_move(*right_move);

            left_move_text.cmp(&right_move_text)
        });

        // Choose from a deterministic generator when Random Seed is set; otherwise use the selector's
        // thread-local generator. choose returns a borrowed move, and copied makes it owned.

        let selected_move = match optional_seeded_random_number_generator(
            configuration.random_seed.as_deref(),
            "random-baseline",
        ) {
            Some(mut random_number_generator) => {
                legal_moves.choose(&mut random_number_generator).copied()
            }
            None => legal_moves
                .choose(&mut self.random_number_generator)
                .copied(),
        };

        // Wrap the optional move in the shared selector result shape used by every strategy.

        MoveSelection::from_move(selected_move)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: fixed_seed_random_selector_is_repeatable
    //
    // Description:
    //
    //   Verify the random baseline can be reproduced for validation runs.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn fixed_seed_random_selector_is_repeatable() {

        // Use two selector instances so repeatability comes from the configured seed, not shared state.

        let position = Position::startpos();
        let limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let mut first_selector = RandomMoveSelector::new();
        let mut second_selector = RandomMoveSelector::new();

        configuration.random_seed = Some("repeatable random baseline".to_string());

        // Run the same seeded selection twice from the same position and limits.

        let control = SearchControl::from_limits(&limits);
        let first_selection =
            first_selector.select_move(&position, &limits, &configuration, &control);
        let second_selection =
            second_selector.select_move(&position, &limits, &configuration, &control);

        // Verify both selectors choose the same legal move.

        assert_eq!(
            first_selection.selected_move,
            second_selection.selected_move
        );
        assert!(position
            .legal_moves()
            .contains(&first_selection.selected_move.unwrap()));
    }
}

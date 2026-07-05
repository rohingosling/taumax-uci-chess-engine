//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Strategy dispatcher for Taumax move selection.
//
//----------------------------------------------------------------------------------------------------------------------

use crate::board::position::Position;
use crate::engine::configuration::{EngineConfiguration, EngineStrategy};
use crate::engine::random::RandomMoveSelector;
use crate::engine::relative::selector::RelativeCausalEntropySelector;
use crate::engine::selector::{MoveSelection, MoveSelector};
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Struct: TaumaxMoveSelector
//
// Description:
//
//   Dispatches configured Taumax strategies to concrete selector implementations.
//
//----------------------------------------------------------------------------------------------------------------------

pub struct TaumaxMoveSelector {
    random_move_selector: RandomMoveSelector,
    relative_causal_entropy_selector: RelativeCausalEntropySelector,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: TaumaxMoveSelector
//
// Description:
//
//   Provides construction behavior for the strategy dispatcher.
//
//----------------------------------------------------------------------------------------------------------------------

impl TaumaxMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a Taumax move selector with active strategy implementations.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new() -> Self {

        // Keep both strategy implementations alive so switching Strategy is a cheap dispatch decision.

        Self {
            random_move_selector: RandomMoveSelector::new(),
            relative_causal_entropy_selector: RelativeCausalEntropySelector::new(),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for TaumaxMoveSelector
//
// Description:
//
//   Uses the normal constructor as the selector default.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for TaumaxMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Create the default Taumax move selector.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {
        Self::new()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: MoveSelector for TaumaxMoveSelector
//
// Description:
//
//   Selects a move using the configured Taumax strategy.
//
//----------------------------------------------------------------------------------------------------------------------

impl MoveSelector for TaumaxMoveSelector {

    //------------------------------------------------------------------------------------------------------------------
    // Method: select_move
    //
    // Description:
    //
    //   Dispatch move selection to the active strategy.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn select_move(
        &mut self,
        position: &Position,
        limits: &SearchLimits,
        configuration: &EngineConfiguration,
        control: &SearchControl,
    ) -> MoveSelection {

        // Dispatch on the typed configuration value instead of comparing UCI strings during search.

        match configuration.strategy {
            EngineStrategy::Random => {
                self.random_move_selector
                    .select_move(position, limits, configuration, control)
            }
            EngineStrategy::RelativeCausalEntropy => self
                .relative_causal_entropy_selector
                .select_move(position, limits, configuration, control),
        }
    }
}

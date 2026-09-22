//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Effective Taumax horizon calculation.
//
//----------------------------------------------------------------------------------------------------------------------

use crate::engine::configuration::EngineConfiguration;
use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Function: effective_taumax_depth
//
// Description:
//
//   Compute the bounded future horizon for one go command.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn effective_taumax_depth(configuration: &EngineConfiguration, limits: &SearchLimits) -> u64 {

    // Compute the horizon as the engine's configured Max Depth, optionally capped by the go command.

    match limits.depth {
        Some(command_depth) => configuration.max_depth.min(command_depth),
        None => configuration.max_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: depth_without_go_depth_uses_configured_tau_depth
    //
    // Description:
    //
    //   Verify Max Depth is used when the GUI supplies no go-depth cap.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn depth_without_go_depth_uses_configured_tau_depth() {

        // Build limits with no go-depth field so the configured engine depth is the only cap.

        let mut configuration = EngineConfiguration::default();
        let limits = SearchLimits::default();

        configuration.max_depth = 7;

        assert_eq!(effective_taumax_depth(&configuration, &limits), 7);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: depth_with_go_depth_uses_minimum_of_configured_and_command_depth
    //
    // Description:
    //
    //   Verify the effective horizon follows min(Max Depth, go depth).
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn depth_with_go_depth_uses_minimum_of_configured_and_command_depth() {

        // Compare both sides of min(Max Depth, go depth): first the command is smaller, then larger.

        let mut configuration = EngineConfiguration::default();
        let mut limits = SearchLimits::default();

        configuration.max_depth = 6;
        limits.depth = Some(4);

        assert_eq!(effective_taumax_depth(&configuration, &limits), 4);

        limits.depth = Some(9);

        assert_eq!(effective_taumax_depth(&configuration, &limits), 6);
    }
}

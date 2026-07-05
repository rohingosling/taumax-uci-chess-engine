//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Library entry point for the Taumax UCI engine.
//
//----------------------------------------------------------------------------------------------------------------------

// Export the board adapter, engine state, search controls, and UCI protocol layers as the public
// library surface used by the binary and integration tests.

pub mod board;
pub mod engine;
pub mod search;
pub mod uci;

// Keep the identity strings centralized so the UCI response layer and tests share the same source.

pub const ENGINE_NAME: &str = "Taumax";
pub const ENGINE_AUTHOR: &str = "Rohin Gosling";

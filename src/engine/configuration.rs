//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Typed configuration for UCI-controlled Taumax engine options.
//
//----------------------------------------------------------------------------------------------------------------------

use std::error::Error;
use std::fmt;

pub const STRATEGY_OPTION_NAME: &str = "Strategy";
pub const MAX_DEPTH_OPTION_NAME: &str = "Max Depth";
pub const RANDOM_SEED_OPTION_NAME: &str = "Random Seed";
pub const DIAGNOSTICS_TRACE_OPTION_NAME: &str = "Diagnostics Trace";
pub const GPU_OPTION_NAME: &str = "GPU Acceleration";

// Max Depth is a UCI spin option, so the engine advertises a bounded integer range to the GUI.

pub const MAX_DEPTH_DEFAULT: u64 = 6;
pub const MAX_DEPTH_MINIMUM: u64 = 1;
pub const MAX_DEPTH_MAXIMUM: u64 = 12;

//----------------------------------------------------------------------------------------------------------------------
// Enum: EngineStrategy
//
// Description:
//
//   Identifies the configured Taumax move-selection strategy.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EngineStrategy {
    #[default]
    Random,
    RelativeCausalEntropy,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: EngineConfiguration
//
// Description:
//
//   Stores persistent engine settings controlled by UCI setoption commands.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineConfiguration {
    pub strategy: EngineStrategy,
    pub max_depth: u64,
    pub random_seed: Option<String>,
    pub diagnostics_trace: bool,
    pub gpu: bool,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: OptionUpdateStatus
//
// Description:
//
//   Reports whether a setoption command updated a known option or named an unknown option.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptionUpdateStatus {
    Updated,
    Unknown,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: EngineConfigurationError
//
// Description:
//
//   Represents invalid values supplied for known engine options.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineConfigurationError {
    MissingValue {
        option_name: String,
    },
    InvalidValue {
        option_name: String,
        value: String,
    },
    ValueOutOfRange {
        option_name: String,
        value: u64,
        minimum: u64,
        maximum: u64,
    },
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: EngineStrategy
//
// Description:
//
//   Provides UCI conversion helpers for strategy values.
//
//----------------------------------------------------------------------------------------------------------------------

impl EngineStrategy {
    pub const VALUES: [Self; 2] = [Self::Random, Self::RelativeCausalEntropy];

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: as_uci_value
    //
    // Description:
    //
    //   Return the canonical UCI option value for this strategy.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn as_uci_value(self) -> &'static str {

        // Keep strategy names stable because GUIs persist option values by their exact text.

        match self {
            Self::Random => "Random",
            Self::RelativeCausalEntropy => "RelativeCausalEntropy",
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: parse_uci_value
    //
    // Description:
    //
    //   Parse a UCI option value into a strategy.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn parse_uci_value(value: &str) -> Option<Self> {

        // Compare case-insensitively so minor GUI casing differences do not reject a valid strategy.

        Self::VALUES
            .into_iter()
            .find(|strategy| strategy.as_uci_value().eq_ignore_ascii_case(value))
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Display for EngineStrategy
//
// Description:
//
//   Formats strategy values for UCI text.
//
//----------------------------------------------------------------------------------------------------------------------

impl fmt::Display for EngineStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {

        // Display uses the canonical UCI value so formatting and option advertisement stay aligned.

        formatter.write_str(self.as_uci_value())
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for EngineConfiguration
//
// Description:
//
//   Creates the default Taumax option set.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for EngineConfiguration {
    fn default() -> Self {

        // Defaults intentionally match the advertised UCI option defaults.

        Self {
            strategy: EngineStrategy::default(),
            max_depth: MAX_DEPTH_DEFAULT,
            random_seed: None,
            diagnostics_trace: false,
            gpu: false,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: EngineConfiguration
//
// Description:
//
//   Applies UCI setoption updates to the typed configuration.
//
//----------------------------------------------------------------------------------------------------------------------

impl EngineConfiguration {

    //------------------------------------------------------------------------------------------------------------------
    // Method: set_option
    //
    // Description:
    //
    //   Update a known option from UCI text, or report that the option name is unknown.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn set_option(
        &mut self,
        option_name: &str,
        value: Option<&str>,
    ) -> Result<OptionUpdateStatus, EngineConfigurationError> {

        // Normalize the name once so user-facing option names can contain spaces while matching stays
        // simple and case-insensitive.

        let normalized_option_name = option_name.trim().to_ascii_lowercase();

        match normalized_option_name.as_str() {
            "strategy" => {

                // Parse the combo value before mutating the configuration, so invalid values leave
                // the previous strategy intact.

                self.strategy = parse_strategy_option(STRATEGY_OPTION_NAME, value)?;
                Ok(OptionUpdateStatus::Updated)
            }
            "max depth" => {

                // Compute the bounded spin value and reject values outside the advertised range.

                self.max_depth = parse_spin_option(
                    MAX_DEPTH_OPTION_NAME,
                    value,
                    MAX_DEPTH_MINIMUM,
                    MAX_DEPTH_MAXIMUM,
                )?;
                Ok(OptionUpdateStatus::Updated)
            }
            "random seed" => {

                // Empty seed text means "use normal nondeterministic randomness" rather than an
                // empty deterministic seed.

                self.random_seed = value
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_string);
                Ok(OptionUpdateStatus::Updated)
            }
            "diagnostics trace" => {

                // UCI check options are textual booleans, not native bool values.

                self.diagnostics_trace = parse_check_option(DIAGNOSTICS_TRACE_OPTION_NAME, value)?;
                Ok(OptionUpdateStatus::Updated)
            }
            "gpu acceleration" => {

                // The GPU option records a request; runtime availability is decided later by the
                // relative leaf-evaluation backend.

                self.gpu = parse_check_option(GPU_OPTION_NAME, value)?;
                Ok(OptionUpdateStatus::Updated)
            }
            _ => {

                // Unknown options are ignored for UCI compatibility with GUIs that send common
                // engine settings this engine does not implement.

                Ok(OptionUpdateStatus::Unknown)
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: parse_strategy_option
//
// Description:
//
//   Parse a Strategy option value.
//
//----------------------------------------------------------------------------------------------------------------------

fn parse_strategy_option(
    option_name: &str,
    value: Option<&str>,
) -> Result<EngineStrategy, EngineConfigurationError> {

    // Require a non-empty value before attempting to map it onto a known strategy.

    let value_text = required_value(option_name, value)?;

    EngineStrategy::parse_uci_value(value_text).ok_or_else(|| {
        EngineConfigurationError::InvalidValue {
            option_name: option_name.to_string(),
            value: value_text.to_string(),
        }
    })
}

//----------------------------------------------------------------------------------------------------------------------
// Function: parse_spin_option
//
// Description:
//
//   Parse a bounded numeric spin option value.
//
//----------------------------------------------------------------------------------------------------------------------

fn parse_spin_option(
    option_name: &str,
    value: Option<&str>,
    minimum: u64,
    maximum: u64,
) -> Result<u64, EngineConfigurationError> {

    // Parse the text as an unsigned integer because UCI spin values arrive as strings.

    let value_text = required_value(option_name, value)?;
    let parsed_value =
        value_text
            .parse::<u64>()
            .map_err(|_| EngineConfigurationError::InvalidValue {
                option_name: option_name.to_string(),
                value: value_text.to_string(),
            })?;

    if parsed_value < minimum || parsed_value > maximum {

        // Preserve the parsed number in the error so diagnostics can show both the bad value and the
        // legal range.

        return Err(EngineConfigurationError::ValueOutOfRange {
            option_name: option_name.to_string(),
            value: parsed_value,
            minimum,
            maximum,
        });
    }

    Ok(parsed_value)
}

//----------------------------------------------------------------------------------------------------------------------
// Function: parse_check_option
//
// Description:
//
//   Parse a boolean UCI check option value.
//
//----------------------------------------------------------------------------------------------------------------------

fn parse_check_option(
    option_name: &str,
    value: Option<&str>,
) -> Result<bool, EngineConfigurationError> {

    // UCI check options accept exactly "true" or "false" after case normalization.

    let value_text = required_value(option_name, value)?;

    match value_text.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(EngineConfigurationError::InvalidValue {
            option_name: option_name.to_string(),
            value: value_text.to_string(),
        }),
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: required_value
//
// Description:
//
//   Return a non-empty option value or a MissingValue error.
//
//----------------------------------------------------------------------------------------------------------------------

fn required_value<'a>(
    option_name: &str,
    value: Option<&'a str>,
) -> Result<&'a str, EngineConfigurationError> {

    // Trim whitespace before validation so " value true " behaves the same as "value true".

    value
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| EngineConfigurationError::MissingValue {
            option_name: option_name.to_string(),
        })
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Display for EngineConfigurationError
//
// Description:
//
//   Formats configuration errors as human-readable diagnostics.
//
//----------------------------------------------------------------------------------------------------------------------

impl fmt::Display for EngineConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {

        // Format errors for stderr diagnostics; stdout must remain reserved for UCI protocol output.

        match self {
            Self::MissingValue { option_name } => {
                write!(formatter, "missing value for option '{option_name}'")
            }
            Self::InvalidValue { option_name, value } => {
                write!(
                    formatter,
                    "invalid value '{value}' for option '{option_name}'"
                )
            }
            Self::ValueOutOfRange {
                option_name,
                value,
                minimum,
                maximum,
            } => {
                write!(
                    formatter,
                    "value {value} for option '{option_name}' is outside range {minimum}..={maximum}"
                )
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Error for EngineConfigurationError
//
// Description:
//
//   Marks configuration errors as standard Rust errors.
//
//----------------------------------------------------------------------------------------------------------------------

impl Error for EngineConfigurationError {}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: defaults_match_advertised_option_values
    //
    // Description:
    //
    //   Verify the typed configuration starts with the advertised default values.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn defaults_match_advertised_option_values() {

        // Construct a fresh configuration and compare every public option against its default.

        let configuration = EngineConfiguration::default();

        assert_eq!(configuration.strategy, EngineStrategy::Random);
        assert_eq!(configuration.max_depth, 6);
        assert_eq!(configuration.random_seed, None);
        assert!(!configuration.diagnostics_trace);
        assert!(!configuration.gpu);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: set_option_updates_known_values
    //
    // Description:
    //
    //   Verify known options update the typed configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn set_option_updates_known_values() {

        // Apply each known option through the same textual path used by the UCI driver.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(STRATEGY_OPTION_NAME, Some("RelativeCausalEntropy")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(
            configuration.set_option(MAX_DEPTH_OPTION_NAME, Some("5")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(
            configuration.set_option(RANDOM_SEED_OPTION_NAME, Some("fixed seed")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(
            configuration.set_option(DIAGNOSTICS_TRACE_OPTION_NAME, Some("true")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(
            configuration.set_option(GPU_OPTION_NAME, Some("true")),
            Ok(OptionUpdateStatus::Updated)
        );

        // Verify the stored typed values after all successful updates have been applied.

        assert_eq!(
            configuration.strategy,
            EngineStrategy::RelativeCausalEntropy
        );
        assert_eq!(configuration.max_depth, 5);
        assert_eq!(configuration.random_seed, Some("fixed seed".to_string()));
        assert!(configuration.diagnostics_trace);
        assert!(configuration.gpu);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: strategy_option_accepts_random_baseline
    //
    // Description:
    //
    //   Verify the random control condition is selectable through EngineStrategy.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn strategy_option_accepts_random_baseline() {

        // Random remains a selectable control strategy for reproducible comparison runs.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(STRATEGY_OPTION_NAME, Some("Random")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(configuration.strategy, EngineStrategy::Random);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: strategy_option_accepts_relative_causal_entropy
    //
    // Description:
    //
    //   Verify the relative causal-entropy contract value is selectable through EngineStrategy.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn strategy_option_accepts_relative_causal_entropy() {

        // RelativeCausalEntropy is the active causal-entropic strategy value exposed to GUIs.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(STRATEGY_OPTION_NAME, Some("RelativeCausalEntropy")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert_eq!(
            configuration.strategy,
            EngineStrategy::RelativeCausalEntropy
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: retired_strategy_values_are_rejected
    //
    // Description:
    //
    //   Verify retired neutral strategy values are invalid for the active Strategy option.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn retired_strategy_values_are_rejected() {

        // A retired value must fail without moving the configuration away from Random.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(STRATEGY_OPTION_NAME, Some("UniformRolloutPathEntropy")),
            Err(EngineConfigurationError::InvalidValue {
                option_name: STRATEGY_OPTION_NAME.to_string(),
                value: "UniformRolloutPathEntropy".to_string(),
            })
        );
        assert_eq!(configuration.strategy, EngineStrategy::Random);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: retired_option_names_are_unknown
    //
    // Description:
    //
    //   Verify retired neutral option names are ignored as unknown GUI options.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn retired_option_names_are_unknown() {

        // Old option names are treated like any other unrecognized GUI option.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option("TauSamples", Some("512")),
            Ok(OptionUpdateStatus::Unknown)
        );
        assert_eq!(
            configuration.set_option("TauMacrostate", Some("path")),
            Ok(OptionUpdateStatus::Unknown)
        );
        assert_eq!(configuration, EngineConfiguration::default());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: gpu_option_is_known
    //
    // Description:
    //
    //   Verify the GPU request option updates typed configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn gpu_option_is_known() {

        // The user-facing executable always exposes the GPU request flag.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(GPU_OPTION_NAME, Some("true")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert!(configuration.gpu);
        assert_eq!(
            configuration.set_option(GPU_OPTION_NAME, Some("false")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert!(!configuration.gpu);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: unknown_option_is_reported_without_changing_configuration
    //
    // Description:
    //
    //   Verify unknown GUI options are ignored for compatibility.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn unknown_option_is_reported_without_changing_configuration() {

        // Hash is a common GUI option, but this engine has no hash-table setting yet.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option("Hash", Some("128")),
            Ok(OptionUpdateStatus::Unknown)
        );
        assert_eq!(configuration, EngineConfiguration::default());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: invalid_value_keeps_previous_value
    //
    // Description:
    //
    //   Verify malformed values do not partially update the configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn invalid_value_keeps_previous_value() {

        // First set a valid value so the malformed update has a prior value it must preserve.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(MAX_DEPTH_OPTION_NAME, Some("5")),
            Ok(OptionUpdateStatus::Updated)
        );
        assert!(configuration
            .set_option(MAX_DEPTH_OPTION_NAME, Some("not-a-number"))
            .is_err());
        assert_eq!(configuration.max_depth, 5);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: out_of_range_spin_value_is_rejected
    //
    // Description:
    //
    //   Verify spin options enforce their advertised limits.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn out_of_range_spin_value_is_rejected() {

        // Max Depth 13 is one step above the advertised upper bound.

        let mut configuration = EngineConfiguration::default();

        assert_eq!(
            configuration.set_option(MAX_DEPTH_OPTION_NAME, Some("13")),
            Err(EngineConfigurationError::ValueOutOfRange {
                option_name: MAX_DEPTH_OPTION_NAME.to_string(),
                value: 13,
                minimum: MAX_DEPTH_MINIMUM,
                maximum: MAX_DEPTH_MAXIMUM,
            })
        );
        assert_eq!(configuration.max_depth, MAX_DEPTH_DEFAULT);
    }
}

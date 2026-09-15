//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Root-move score data and UCI trace formatting.
//
//----------------------------------------------------------------------------------------------------------------------

use crate::engine::configuration::EngineStrategy;

//----------------------------------------------------------------------------------------------------------------------
// Struct: RootMoveTraceField
//
// Description:
//
//   Stores one strategy-specific root-move trace field.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootMoveTraceField {
    pub name: String,
    pub value: String,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RootMoveScore
//
// Description:
//
//   Stores the score and metadata for one root move.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct RootMoveScore {
    pub strategy: EngineStrategy,
    pub move_text: String,
    pub score: f64,
    pub depth: u64,
    pub fields: Vec<RootMoveTraceField>,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RootMoveTraceField
//
// Description:
//
//   Provides construction behavior for trace fields.
//
//----------------------------------------------------------------------------------------------------------------------

impl RootMoveTraceField {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create one named trace field.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {

        // Convert both inputs once so trace fields own their text independently of the caller.

        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RootMoveScore
//
// Description:
//
//   Provides construction and trace formatting for root-move scores.
//
//----------------------------------------------------------------------------------------------------------------------

impl RootMoveScore {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create root-move score data with no strategy-specific fields.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(
        strategy: EngineStrategy,
        move_text: impl Into<String>,
        score: f64,
        depth: u64,
    ) -> Self {

        // Store the common score fields first; strategy-specific fields can be appended later.

        Self {
            strategy,
            move_text: move_text.into(),
            score,
            depth,
            fields: Vec::new(),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: with_field
    //
    // Description:
    //
    //   Attach a strategy-specific trace field to the score.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_field(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {

        // Preserve insertion order because trace output is easier to read when fields appear
        // consistently from general to specific.

        self.fields.push(RootMoveTraceField::new(name, value));
        self
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: to_trace_line
    //
    // Description:
    //
    //   Format the score as a valid UCI info string trace line.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn to_trace_line(&self) -> String {

        // Start with the UCI-safe prefix and the fields common to every strategy.

        let mut fields = vec![
            "info string tau".to_string(),
            format!("strategy={}", self.strategy),
            format!("move={}", self.move_text),
            format!("score={:.6}", self.score),
            format!("depth={}", self.depth),
        ];

        // Append strategy-specific key-value pairs without whitespace inside a field.

        for field in &self.fields {
            fields.push(format!("{}={}", field.name, field.value));
        }

        // Join fields with single spaces because UCI info strings are whitespace-tokenized by GUIs.

        fields.join(" ")
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: trace_line
//
// Description:
//
//   Format root-move score data as a valid UCI info string trace line.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn trace_line(root_move_score: &RootMoveScore) -> String {

    // Keep the free function as a small adapter for code that formats borrowed root scores.

    root_move_score.to_trace_line()
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: trace_line_formats_root_move_score
    //
    // Description:
    //
    //   Verify trace formatting produces valid UCI info string output.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn trace_line_formats_root_move_score() {

        // Build a score with common fields and two extra fields to exercise both formatting paths.

        let root_move_score = RootMoveScore::new(EngineStrategy::Random, "e2e4", 5.5, 4)
            .with_field("future", "312")
            .with_field("note", "relative");

        // Verify the full trace is a single UCI info string with deterministic field order.

        assert_eq!(
            root_move_score.to_trace_line(),
            "info string tau strategy=Random move=e2e4 score=5.500000 depth=4 future=312 note=relative"
        );

        // Verify the helper function delegates to the same formatting implementation.

        assert_eq!(
            trace_line(&root_move_score),
            root_move_score.to_trace_line()
        );
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Cooperative search cancellation and deadline checks.
//
//----------------------------------------------------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Struct: SearchControl
//
// Description:
//
//   Stores the cancellation state for one search.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SearchControl {
    stop_requested: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: SearchControl
//
// Description:
//
//   Provides cancellation checks used by selectors.
//
//----------------------------------------------------------------------------------------------------------------------

impl SearchControl {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a search-control object from a stop flag and optional deadline.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(stop_requested: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {

        // Store both stop mechanisms together so selectors only need to call should_stop().

        Self {
            stop_requested,
            deadline,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_limits
    //
    // Description:
    //
    //   Create search control from UCI limits using a fresh stop flag.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_limits(limits: &SearchLimits) -> Self {

        // Tests and internal callers use a fresh flag when there is no UCI input thread to share one.

        Self::new(
            Arc::new(AtomicBool::new(false)),
            deadline_from_limits(limits),
        )
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_limits_and_stop_flag
    //
    // Description:
    //
    //   Create search control from UCI limits and a shared stop flag.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_limits_and_stop_flag(
        limits: &SearchLimits,
        stop_requested: Arc<AtomicBool>,
    ) -> Self {

        // The UCI driver passes the process-level stop flag so input can interrupt the active search.

        Self::new(stop_requested, deadline_from_limits(limits))
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: should_stop
    //
    // Description:
    //
    //   Return whether the current search should stop as soon as practical.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn should_stop(&self) -> bool {

        // Stop if either an explicit stop request arrived or the movetime deadline has passed.

        self.stop_requested.load(Ordering::Relaxed)
            || self
                .deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(false)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: request_stop
    //
    // Description:
    //
    //   Request cooperative cancellation.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn request_stop(&self) {

        // Set the flag cooperatively; search code observes it at safe polling points.

        self.stop_requested.store(true, Ordering::Relaxed);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: clear_stop
    //
    // Description:
    //
    //   Clear a previous stop request before starting a new search.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn clear_stop(&self) {

        // Clear the same shared flag so a previous search does not poison the next one.

        self.stop_requested.store(false, Ordering::Relaxed);
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: deadline_from_limits
//
// Description:
//
//   Return the deadline implied by a UCI movetime limit.
//
//----------------------------------------------------------------------------------------------------------------------

fn deadline_from_limits(limits: &SearchLimits) -> Option<Instant> {

    // Compute an absolute deadline only from movetime; other UCI clock fields are stored but unused.

    limits.move_time.map(|move_time| {
        let move_duration = Duration::from_millis(move_time);

        // Add the requested duration to the current instant to get a monotonic stop point.

        Instant::now() + move_duration
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: stop_request_is_observed
    //
    // Description:
    //
    //   Verify explicit cancellation is visible to search code.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn stop_request_is_observed() {

        // Start with a fresh control object whose stop flag is clear.

        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        assert!(!control.should_stop());

        // Request cancellation and verify the same control object observes it.

        control.request_stop();

        assert!(control.should_stop());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: zero_movetime_stops_immediately
    //
    // Description:
    //
    //   Verify a zero-millisecond movetime produces an immediate deadline.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn zero_movetime_stops_immediately() {

        // A zero-duration movetime computes a deadline at the current instant.

        let limits = SearchLimits {
            move_time: Some(0),
            ..Default::default()
        };

        assert!(SearchControl::from_limits(&limits).should_stop());
    }
}

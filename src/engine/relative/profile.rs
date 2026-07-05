//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Profiling and acceleration diagnostics for relative causal entropy.
//
//----------------------------------------------------------------------------------------------------------------------

use std::time::Duration;

use crate::engine::relative::leaf::RelativeLeafEvaluationBackend;

// Profiling thresholds are deliberately conservative: they identify workloads large enough to amortize
// thread scheduling or GPU dispatch overhead.

pub const MAX_PARALLEL_ROOT_WORKERS: usize = 4;
pub const PARALLEL_ROOT_MOVE_THRESHOLD: usize = 4;
pub const GPU_CANDIDATE_MINIMUM_LEAF_COUNT: usize = 16_384;
pub const GPU_CANDIDATE_MINIMUM_BATCH_SIZE: usize = 512;
//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeAccelerationBackend
//
// Description:
//
//   Identifies the relative-search execution path used for one move search.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativeAccelerationBackend {
    SerialCpu,
    ParallelCpu,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeAccelerationBackend
//
// Description:
//
//   Provides trace formatting for acceleration backend labels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeAccelerationBackend {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: trace_value
    //
    // Description:
    //
    //   Return the UCI-safe backend label used in profile traces.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn trace_value(self) -> &'static str {

        // Keep labels compact and stable because Diagnostics Trace output is consumed by tests and GUIs.

        match self {
            Self::SerialCpu => "serial-cpu",
            Self::ParallelCpu => "parallel-cpu",
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeGpuAccelerationOpportunity
//
// Description:
//
//   Classifies whether a relative search has enough batched leaf work to justify GPU execution.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativeGpuAccelerationOpportunity {
    NotCandidate,
    Candidate,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeGpuAccelerationOpportunity
//
// Description:
//
//   Provides trace formatting for GPU acceleration opportunity labels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeGpuAccelerationOpportunity {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: trace_value
    //
    // Description:
    //
    //   Return the UCI-safe GPU opportunity label used in profile traces.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn trace_value(self) -> &'static str {

        // Report candidacy as yes/no so the profile line stays easy to scan in UCI logs.

        match self {
            Self::NotCandidate => "no",
            Self::Candidate => "yes",
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeGpuAccelerationRequest
//
// Description:
//
//   Reports whether a relative search requested the optional GPU leaf backend.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativeGpuAccelerationRequest {
    NotRequested,
    Requested,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeGpuAccelerationRequest
//
// Description:
//
//   Provides trace formatting for GPU request labels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeGpuAccelerationRequest {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: trace_value
    //
    // Description:
    //
    //   Return the UCI-safe GPU request label used in profile traces.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn trace_value(self) -> &'static str {

        // Report the user's GPU request separately from whether a GPU batch actually executed.

        match self {
            Self::NotRequested => "no",
            Self::Requested => "yes",
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeSearchProfileCounters
//
// Description:
//
//   Carries named profile counters before duration conversion.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelativeSearchProfileCounters {
    pub backend: RelativeAccelerationBackend,
    pub leaf_evaluation_backend: RelativeLeafEvaluationBackend,
    pub gpu_acceleration_request: RelativeGpuAccelerationRequest,
    pub root_move_count: usize,
    pub scored_root_move_count: usize,
    pub worker_count: usize,
    pub visited_node_count: usize,
    pub future_leaf_count: usize,
    pub terminal_leaf_count: usize,
    pub leaf_evaluation_batch_count: usize,
    pub largest_leaf_evaluation_batch_size: usize,
    pub elapsed_duration: Duration,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeSearchProfile
//
// Description:
//
//   Stores aggregated work counters for one relative search.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelativeSearchProfile {
    pub backend: RelativeAccelerationBackend,
    pub leaf_evaluation_backend: RelativeLeafEvaluationBackend,
    pub gpu_acceleration_request: RelativeGpuAccelerationRequest,
    pub root_move_count: usize,
    pub scored_root_move_count: usize,
    pub worker_count: usize,
    pub visited_node_count: usize,
    pub future_leaf_count: usize,
    pub terminal_leaf_count: usize,
    pub leaf_evaluation_batch_count: usize,
    pub largest_leaf_evaluation_batch_size: usize,
    pub gpu_acceleration_opportunity: RelativeGpuAccelerationOpportunity,
    pub elapsed_microseconds: u128,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeSearchProfile
//
// Description:
//
//   Provides profile construction and UCI-safe formatting.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeSearchProfile {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_counters
    //
    // Description:
    //
    //   Create a relative-search profile from named collected counters.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn from_counters(counters: RelativeSearchProfileCounters) -> Self {

        // Compute derived profile fields from raw counters at the end of a search.

        Self {
            backend: counters.backend,
            leaf_evaluation_backend: counters.leaf_evaluation_backend,
            gpu_acceleration_request: counters.gpu_acceleration_request,
            root_move_count: counters.root_move_count,
            scored_root_move_count: counters.scored_root_move_count,
            worker_count: counters.worker_count,
            visited_node_count: counters.visited_node_count,
            future_leaf_count: counters.future_leaf_count,
            terminal_leaf_count: counters.terminal_leaf_count,
            leaf_evaluation_batch_count: counters.leaf_evaluation_batch_count,
            largest_leaf_evaluation_batch_size: counters.largest_leaf_evaluation_batch_size,

            // Compute GPU candidacy from both total leaf volume and largest observed batch size.

            gpu_acceleration_opportunity: relative_gpu_acceleration_opportunity(
                counters.future_leaf_count,
                counters.largest_leaf_evaluation_batch_size,
            ),

            // Convert duration to microseconds so trace values remain integer and UCI-safe.

            elapsed_microseconds: counters.elapsed_duration.as_micros(),
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: to_trace_line
    //
    // Description:
    //
    //   Format the profile as a UCI-safe diagnostic trace line.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn to_trace_line(&self) -> String {

        // Build one whitespace-tokenized UCI info string from stable key-value fields.

        format!(
            "info string tau profile strategy=RelativeCausalEntropy acceleration={} leafBackend={} gpuRequested={} roots={} scored={} workers={} nodes={} leaves={} leafBatches={} leafBatchMax={} gpuCandidate={} terminals={} micros={}",
            self.backend.trace_value(),
            self.leaf_evaluation_backend.trace_value(),
            self.gpu_acceleration_request.trace_value(),
            self.root_move_count,
            self.scored_root_move_count,
            self.worker_count,
            self.visited_node_count,
            self.future_leaf_count,
            self.leaf_evaluation_batch_count,
            self.largest_leaf_evaluation_batch_size,
            self.gpu_acceleration_opportunity.trace_value(),
            self.terminal_leaf_count,
            self.elapsed_microseconds
        )
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_gpu_acceleration_opportunity
//
// Description:
//
//   Return whether a relative search has enough batched leaf work to justify trying a GPU backend.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn relative_gpu_acceleration_opportunity(
    future_leaf_count: usize,
    largest_leaf_evaluation_batch_size: usize,
) -> RelativeGpuAccelerationOpportunity {

    // Require both enough total leaf work and enough batch size to make GPU dispatch worthwhile.

    if future_leaf_count >= GPU_CANDIDATE_MINIMUM_LEAF_COUNT
        && largest_leaf_evaluation_batch_size >= GPU_CANDIDATE_MINIMUM_BATCH_SIZE
    {
        RelativeGpuAccelerationOpportunity::Candidate
    } else {
        RelativeGpuAccelerationOpportunity::NotCandidate
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: relative_worker_count_for_root_moves
//
// Description:
//
//   Return the number of worker threads worth using for a root-move batch.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn relative_worker_count_for_root_moves(root_move_count: usize) -> usize {

    // Ask the operating system how many hardware threads are available, falling back to one.

    let available_worker_count = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);

    // Compute the worker count as the minimum of machine capacity, policy cap, and root count.

    available_worker_count
        .min(MAX_PARALLEL_ROOT_WORKERS)
        .min(root_move_count.max(1))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: should_score_roots_in_parallel
//
// Description:
//
//   Return whether a root batch is large enough for parallel CPU scoring.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn should_score_roots_in_parallel(root_move_count: usize, worker_count: usize) -> bool {

    // Parallel root scoring needs enough roots and more than one worker to pay for thread overhead.

    root_move_count >= PARALLEL_ROOT_MOVE_THRESHOLD && worker_count > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    //------------------------------------------------------------------------------------------------------------------
    // Function: profile_trace_line_is_uci_safe
    //
    // Description:
    //
    //   Verify profile diagnostics use whitespace-separated key-value fields.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn profile_trace_line_is_uci_safe() {

        // Build a profile with non-zero counters so every formatted field appears in the output.

        let profile = RelativeSearchProfile::from_counters(RelativeSearchProfileCounters {
            backend: RelativeAccelerationBackend::ParallelCpu,
            leaf_evaluation_backend: RelativeLeafEvaluationBackend::CpuBatch,
            gpu_acceleration_request: RelativeGpuAccelerationRequest::NotRequested,
            root_move_count: 20,
            scored_root_move_count: 20,
            worker_count: 4,
            visited_node_count: 421,
            future_leaf_count: 400,
            terminal_leaf_count: 2,
            leaf_evaluation_batch_count: 5,
            largest_leaf_evaluation_batch_size: 128,
            elapsed_duration: Duration::from_micros(123),
        });

        // Verify the exact trace string because field names and order are part of diagnostics.

        assert_eq!(
            profile.to_trace_line(),
            "info string tau profile strategy=RelativeCausalEntropy acceleration=parallel-cpu leafBackend=cpu-batch gpuRequested=no roots=20 scored=20 workers=4 nodes=421 leaves=400 leafBatches=5 leafBatchMax=128 gpuCandidate=no terminals=2 micros=123"
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: gpu_opportunity_requires_large_leaf_batches
    //
    // Description:
    //
    //   Verify GPU candidacy only appears when both leaf volume and batch size are high enough.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn gpu_opportunity_requires_large_leaf_batches() {

        // Both total leaf count and largest batch size must cross the thresholds.

        assert_eq!(
            relative_gpu_acceleration_opportunity(2_721_274, 1_024),
            RelativeGpuAccelerationOpportunity::Candidate
        );
        assert_eq!(
            relative_gpu_acceleration_opportunity(2_721_274, 128),
            RelativeGpuAccelerationOpportunity::NotCandidate
        );
        assert_eq!(
            relative_gpu_acceleration_opportunity(1_147, 1_024),
            RelativeGpuAccelerationOpportunity::NotCandidate
        );
    }
}

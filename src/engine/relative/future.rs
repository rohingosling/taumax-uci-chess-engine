//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Exact shallow future relative-entropy estimator.
//
//----------------------------------------------------------------------------------------------------------------------

use cozy_chess::{Color, Move};

use crate::board::position::Position;
use crate::engine::configuration::EngineConfiguration;
use crate::engine::horizon::effective_taumax_depth;
use crate::engine::relative::freedom::FreedomPotentialWeights;
use crate::engine::relative::leaf::{
    CpuRelativeLeafEvaluationKernel, RelativeLeafEvaluationBackend, RelativeLeafEvaluationBatcher,
    RelativeLeafEvaluationKernel, DEFAULT_LEAF_EVALUATION_BATCH_SIZE,
};
use crate::engine::relative::terminal::TerminalStatePolicy;
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;

pub const DEFAULT_FUTURE_WEIGHT: f64 = 1.0;
pub const DEFAULT_EXACT_NODE_BUDGET: usize = 4096;

//----------------------------------------------------------------------------------------------------------------------
// Enum: FutureEntropyStatus
//
// Description:
//
//   Reports whether an exact future-entropy estimate completed or degraded gracefully.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FutureEntropyStatus {
    Complete,
    Cancelled,
    NodeBudgetExceeded,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: FutureEntropyStatus
//
// Description:
//
//   Provides status-combination behavior for recursive estimates.
//
//----------------------------------------------------------------------------------------------------------------------

impl FutureEntropyStatus {

    //------------------------------------------------------------------------------------------------------------------
    // Function: combine
    //
    // Description:
    //
    //   Preserve the most limiting status seen while aggregating child estimates.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub(crate) fn combine(self, next_status: Self) -> Self {

        // Keep cancellation as the strongest status, then node-budget degradation, then complete.

        match (self, next_status) {
            (Self::Cancelled, _) | (_, Self::Cancelled) => Self::Cancelled,
            (Self::NodeBudgetExceeded, _) | (_, Self::NodeBudgetExceeded) => {
                Self::NodeBudgetExceeded
            }
            (Self::Complete, Self::Complete) => Self::Complete,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: FutureEntropyEstimate
//
// Description:
//
//   Contains the result of one future relative-entropy estimate.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct FutureEntropyEstimate {
    pub requested_depth: u64,
    pub status: FutureEntropyStatus,
    pub value: f64,
    pub partition_sum: f64,
    pub leaf_count: usize,
    pub terminal_leaf_count: usize,
    pub leaf_evaluation_backend: RelativeLeafEvaluationBackend,
    pub leaf_evaluation_batch_count: usize,
    pub largest_leaf_evaluation_batch_size: usize,
    pub visited_node_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeFutureEntropyEstimator
//
// Description:
//
//   Computes exact shallow future relative entropy under an internal node budget.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct RelativeFutureEntropyEstimator {
    pub freedom_weights: FreedomPotentialWeights,
    pub future_weight: f64,
    pub node_budget: usize,
    pub terminal_state_policy: TerminalStatePolicy,
    pub leaf_evaluation_batch_size: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for RelativeFutureEntropyEstimator
//
// Description:
//
//   Provides initial internal estimator parameters.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for RelativeFutureEntropyEstimator {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Return the default exact shallow estimator configuration.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {

        // Use default freedom weights, unit future weighting, and the internal exact-node budget.

        Self::new(
            FreedomPotentialWeights::default(),
            DEFAULT_FUTURE_WEIGHT,
            DEFAULT_EXACT_NODE_BUDGET,
        )
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeFutureEntropyEstimator
//
// Description:
//
//   Provides exact expansion and graceful fallback behavior.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeFutureEntropyEstimator {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a future relative-entropy estimator.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(
        freedom_weights: FreedomPotentialWeights,
        future_weight: f64,
        node_budget: usize,
    ) -> Self {

        // Store estimator parameters that stay constant for one future-entropy evaluation.

        Self {
            freedom_weights,
            future_weight,
            node_budget,
            terminal_state_policy: TerminalStatePolicy::default(),
            leaf_evaluation_batch_size: DEFAULT_LEAF_EVALUATION_BATCH_SIZE,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: with_terminal_state_policy
    //
    // Description:
    //
    //   Return this estimator with an overridden terminal boundary policy.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_terminal_state_policy(
        mut self,
        terminal_state_policy: TerminalStatePolicy,
    ) -> Self {

        // Override terminal boundary scoring while preserving the other estimator settings.

        self.terminal_state_policy = terminal_state_policy;

        self
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: with_leaf_evaluation_batch_size
    //
    // Description:
    //
    //   Return this estimator with an overridden non-terminal leaf batch size.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_leaf_evaluation_batch_size(mut self, leaf_evaluation_batch_size: usize) -> Self {

        // Clamp to at least one so non-terminal leaves can always be flushed.

        self.leaf_evaluation_batch_size = leaf_evaluation_batch_size.max(1);

        self
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimate_from_configuration
    //
    // Description:
    //
    //   Estimate future entropy using Max Depth and any go-depth cap.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn estimate_from_configuration(
        &self,
        taumax_position: &Position,
        opponent_position: &Position,
        configuration: &EngineConfiguration,
        limits: &SearchLimits,
        control: &SearchControl,
    ) -> FutureEntropyEstimate {

        // Compute the requested horizon from UCI configuration before running the exact estimator.

        self.estimate(
            taumax_position,
            opponent_position,
            effective_taumax_depth(configuration, limits),
            control,
        )
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimate
    //
    // Description:
    //
    //   Estimate future relative entropy to the requested exact depth.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn estimate(
        &self,
        taumax_position: &Position,
        opponent_position: &Position,
        depth: u64,
        control: &SearchControl,
    ) -> FutureEntropyEstimate {

        // Use the scalar CPU kernel when no explicit leaf backend is supplied.

        self.estimate_with_leaf_kernel(
            taumax_position,
            opponent_position,
            depth,
            control,
            CpuRelativeLeafEvaluationKernel,
        )
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimate_with_leaf_kernel
    //
    // Description:
    //
    //   Estimate future relative entropy with an explicitly supplied leaf-evaluation kernel.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn estimate_with_leaf_kernel<Kernel: RelativeLeafEvaluationKernel>(
        &self,
        taumax_position: &Position,
        opponent_position: &Position,
        depth: u64,
        control: &SearchControl,
        leaf_evaluation_kernel: Kernel,
    ) -> FutureEntropyEstimate {

        // Track visited tree nodes separately from evaluated leaves.

        let mut visited_node_count = 0;

        // Collect non-terminal horizon leaves into batches for the chosen backend kernel.

        let mut leaf_evaluation_batcher = RelativeLeafEvaluationBatcher::with_kernel(
            self.freedom_weights,
            self.future_weight,
            self.leaf_evaluation_batch_size,
            leaf_evaluation_kernel,
        );

        // Seed recursion from Taumax's current side-to-move position.

        let recursive_context = RecursiveEntropyContext {
            current_position: taumax_position,
            taumax_position,
            opponent_position,
            taumax_side: taumax_position.board().side_to_move(),
            remaining_depth: depth,
        };

        // Expand the tree and queue any non-terminal leaves for later batched evaluation.

        let recursive_estimate = self.estimate_recursive(
            recursive_context,
            control,
            &mut visited_node_count,
            &mut leaf_evaluation_batcher,
        );

        // Flush all queued non-terminal leaves and combine them with terminal leaf contributions.

        let leaf_evaluation_summary = leaf_evaluation_batcher.finish();
        let partition_sum =
            recursive_estimate.partition_sum + leaf_evaluation_summary.partition_sum;

        // Non-terminal leaves are counted by the batcher; terminal leaves are counted during recursion.

        debug_assert_eq!(
            leaf_evaluation_summary.leaf_count + recursive_estimate.terminal_leaf_count,
            recursive_estimate.leaf_count
        );

        // Compute the final entropy-like value as ln(partition sum).

        FutureEntropyEstimate {
            requested_depth: depth,
            status: recursive_estimate.status,
            value: partition_sum.ln(),
            partition_sum,
            leaf_count: recursive_estimate.leaf_count,
            terminal_leaf_count: recursive_estimate.terminal_leaf_count,
            leaf_evaluation_backend: leaf_evaluation_summary.backend,
            leaf_evaluation_batch_count: leaf_evaluation_summary.batch_count,
            largest_leaf_evaluation_batch_size: leaf_evaluation_summary.largest_batch_size,
            visited_node_count,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimate_recursive
    //
    // Description:
    //
    //   Recursively expand legal futures in deterministic UCI order.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn estimate_recursive<Kernel: RelativeLeafEvaluationKernel>(
        &self,
        context: RecursiveEntropyContext<'_>,
        control: &SearchControl,
        visited_node_count: &mut usize,
        leaf_evaluation_batcher: &mut RelativeLeafEvaluationBatcher<Kernel>,
    ) -> RecursiveEntropyEstimate {

        // Stop cooperatively before expanding this node if the GUI or deadline requested cancellation.

        if control.should_stop() {
            return self.fallback_estimate(
                context.taumax_position,
                context.opponent_position,
                FutureEntropyStatus::Cancelled,
                leaf_evaluation_batcher,
            );
        }

        // Stop exact expansion when the internal budget is exhausted, but still score a fallback leaf.

        if *visited_node_count >= self.node_budget {
            return self.fallback_estimate(
                context.taumax_position,
                context.opponent_position,
                FutureEntropyStatus::NodeBudgetExceeded,
                leaf_evaluation_batcher,
            );
        }

        // Count this node after the budget and cancellation gates pass.

        *visited_node_count += 1;

        // Terminal states are scored by explicit boundary policy instead of ordinary mobility.

        if context.current_position.terminal_state().is_some() {
            return self.terminal_state_estimate(
                context.current_position,
                context.taumax_position,
                context.opponent_position,
                context.taumax_side,
                leaf_evaluation_batcher,
            );
        }

        // At the horizon, queue one non-terminal leaf for immediate relative-freedom evaluation.

        if context.remaining_depth == 0 {
            return self.fallback_estimate(
                context.taumax_position,
                context.opponent_position,
                FutureEntropyStatus::Complete,
                leaf_evaluation_batcher,
            );
        }

        // Expand legal moves in deterministic UCI order to keep estimates reproducible.

        let legal_moves = sorted_legal_moves(context.current_position);

        // A no-move board should be treated as terminal even if the adapter status is conservative.

        if legal_moves.is_empty() {
            return self.terminal_state_estimate(
                context.current_position,
                context.taumax_position,
                context.opponent_position,
                context.taumax_side,
                leaf_evaluation_batcher,
            );
        }

        // Aggregate all child partition sums and leaf counts into one estimate for this node.

        let mut aggregate_estimate = RecursiveEntropyEstimate::empty();

        for chess_move in legal_moves {

            // Apply the generated legal move to produce the child board.

            let mut next_position = context.current_position.clone();
            next_position
                .apply_move(chess_move)
                .expect("generated legal move must apply cleanly");

            // Alternate the natural Taumax/opponent position pair depending on whose turn moved.

            let child_context =
                if context.current_position.board().side_to_move() == context.taumax_side {
                    RecursiveEntropyContext {
                        current_position: &next_position,
                        taumax_position: context.taumax_position,
                        opponent_position: &next_position,
                        taumax_side: context.taumax_side,
                        remaining_depth: context.remaining_depth - 1,
                    }
                } else {
                    RecursiveEntropyContext {
                        current_position: &next_position,
                        taumax_position: &next_position,
                        opponent_position: context.opponent_position,
                        taumax_side: context.taumax_side,
                        remaining_depth: context.remaining_depth - 1,
                    }
                };

            // Recurse into the child and fold the result into the aggregate estimate.

            let child_estimate = self.estimate_recursive(
                child_context,
                control,
                visited_node_count,
                leaf_evaluation_batcher,
            );
            let child_status = child_estimate.status;
            aggregate_estimate.add_child(child_estimate);

            // Stop expanding siblings after cancellation or budget degradation appears.

            if child_status != FutureEntropyStatus::Complete {
                break;
            }
        }

        aggregate_estimate
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: fallback_estimate
    //
    // Description:
    //
    //   Score one non-terminal fallback leaf by its immediate relative freedom potential.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn fallback_estimate<Kernel: RelativeLeafEvaluationKernel>(
        &self,
        taumax_position: &Position,
        opponent_position: &Position,
        status: FutureEntropyStatus,
        leaf_evaluation_batcher: &mut RelativeLeafEvaluationBatcher<Kernel>,
    ) -> RecursiveEntropyEstimate {

        // Queue the non-terminal leaf; its partition contribution will be computed by the batcher.

        leaf_evaluation_batcher.push_leaf(taumax_position, opponent_position);

        // Return zero direct partition contribution because the batcher owns this leaf's score.

        RecursiveEntropyEstimate {
            status,
            partition_sum: 0.0,
            leaf_count: 1,
            terminal_leaf_count: 0,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: terminal_state_estimate
    //
    // Description:
    //
    //   Score one terminal leaf by explicit win/loss/draw future-agency boundaries.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn terminal_state_estimate<Kernel: RelativeLeafEvaluationKernel>(
        &self,
        current_position: &Position,
        taumax_position: &Position,
        opponent_position: &Position,
        taumax_side: Color,
        leaf_evaluation_batcher: &mut RelativeLeafEvaluationBatcher<Kernel>,
    ) -> RecursiveEntropyEstimate {

        // If the policy recognizes the terminal board, convert its boundary score to a partition term.

        if let Some((_, terminal_score)) = self
            .terminal_state_policy
            .score_position(current_position, taumax_side)
        {

            // Compute the terminal contribution as exp(T * terminal_score).

            let weighted_score = self.future_weight * terminal_score;

            return RecursiveEntropyEstimate {
                status: FutureEntropyStatus::Complete,
                partition_sum: weighted_score.exp(),
                leaf_count: 1,
                terminal_leaf_count: 1,
            };
        }

        // If no terminal score is available, fall back to ordinary non-terminal leaf scoring.

        self.fallback_estimate(
            taumax_position,
            opponent_position,
            FutureEntropyStatus::Complete,
            leaf_evaluation_batcher,
        )
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RecursiveEntropyContext
//
// Description:
//
//   Carries the naturally alternating positions needed for one recursive estimate node.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct RecursiveEntropyContext<'a> {
    current_position: &'a Position,
    taumax_position: &'a Position,
    opponent_position: &'a Position,
    taumax_side: Color,
    remaining_depth: u64,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RecursiveEntropyEstimate
//
// Description:
//
//   Internal accumulator for recursive partition-sum estimates.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct RecursiveEntropyEstimate {
    status: FutureEntropyStatus,
    partition_sum: f64,
    leaf_count: usize,
    terminal_leaf_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RecursiveEntropyEstimate
//
// Description:
//
//   Provides small aggregation helpers for child estimates.
//
//----------------------------------------------------------------------------------------------------------------------

impl RecursiveEntropyEstimate {

    //------------------------------------------------------------------------------------------------------------------
    // Function: empty
    //
    // Description:
    //
    //   Create an empty complete recursive estimate.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn empty() -> Self {

        // Start from a complete aggregate with no partition mass and no leaves.

        Self {
            status: FutureEntropyStatus::Complete,
            partition_sum: 0.0,
            leaf_count: 0,
            terminal_leaf_count: 0,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: add_child
    //
    // Description:
    //
    //   Add one child estimate into this aggregate.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn add_child(&mut self, child_estimate: Self) {

        // Combine status pessimistically and add all child counts and partition mass.

        self.status = self.status.combine(child_estimate.status);
        self.partition_sum += child_estimate.partition_sum;
        self.leaf_count += child_estimate.leaf_count;
        self.terminal_leaf_count += child_estimate.terminal_leaf_count;
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: sorted_legal_moves
//
// Description:
//
//   Return legal moves in deterministic UCI-text order.
//
//----------------------------------------------------------------------------------------------------------------------

fn sorted_legal_moves(position: &Position) -> Vec<Move> {

    // Sort generated legal moves by UCI text so traversal is deterministic.

    let mut legal_moves = position.legal_moves();

    legal_moves.sort_by(|left_move, right_move| {
        let left_move_text = position.display_uci_move(*left_move);
        let right_move_text = position.display_uci_move(*right_move);

        left_move_text.cmp(&right_move_text)
    });

    legal_moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::relative::leaf::GpuRelativeLeafEvaluationKernel;
    use crate::engine::relative::terminal::{
        DEFAULT_TERMINAL_DRAW_SCORE, DEFAULT_TERMINAL_LOSS_SCORE,
    };

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

        // Use a tight tolerance for deterministic CPU estimator checks.

        assert!(
            (left_value - right_value).abs() < FLOAT_TOLERANCE,
            "left={left_value}, right={right_value}"
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: assert_relative_close
    //
    // Description:
    //
    //   Verify two floating-point values are close enough for f64 CPU and f32 GPU parity.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn assert_relative_close(left_value: f64, right_value: f64, tolerance: f64) {

        // Compute scale-normalized error because GPU uses f32 feature math.

        let scale = left_value.abs().max(right_value.abs()).max(1.0);
        let relative_difference = (left_value - right_value).abs() / scale;

        assert!(
            relative_difference <= tolerance,
            "left={left_value}, right={right_value}, relative={relative_difference}"
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: paired_start_positions
    //
    // Description:
    //
    //   Build naturally alternating Taumax/opponent positions for estimator tests.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn paired_start_positions() -> (Position, Position) {

        // Keep Taumax at startpos and give the opponent position Black to move after e2e4.

        let taumax_position = Position::startpos();
        let mut opponent_position = Position::startpos();

        opponent_position.apply_uci_move("e2e4").unwrap();

        (taumax_position, opponent_position)
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: exact_depth_one_fixture_is_deterministic
    //
    // Description:
    //
    //   Verify exact depth-1 expansion has a stable leaf count and entropy value.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn exact_depth_one_fixture_is_deterministic() {

        // Run the same depth-one estimate twice from the same paired positions.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator = RelativeFutureEntropyEstimator::default();
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let first_estimate = estimator.estimate(&taumax_position, &opponent_position, 1, &control);
        let second_estimate = estimator.estimate(&taumax_position, &opponent_position, 1, &control);

        // Depth one from startpos should visit the root plus twenty child leaves.

        assert_eq!(first_estimate.status, FutureEntropyStatus::Complete);
        assert_eq!(first_estimate.leaf_count, 20);
        assert_eq!(first_estimate.terminal_leaf_count, 0);
        assert_eq!(first_estimate.visited_node_count, 21);
        assert_close(first_estimate.value, second_estimate.value);
        assert_close(first_estimate.partition_sum, second_estimate.partition_sum);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: exact_depth_one_records_leaf_evaluation_batches
    //
    // Description:
    //
    //   Verify non-terminal horizon leaves are grouped into deterministic evaluation batches.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn exact_depth_one_records_leaf_evaluation_batches() {

        // Use batch size eight so twenty leaves split into three batches: 8, 8, and 4.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator =
            RelativeFutureEntropyEstimator::default().with_leaf_evaluation_batch_size(8);
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let estimate = estimator.estimate(&taumax_position, &opponent_position, 1, &control);

        // Verify both search leaf counts and backend batching counters.

        assert_eq!(estimate.status, FutureEntropyStatus::Complete);
        assert_eq!(estimate.leaf_count, 20);
        assert_eq!(estimate.terminal_leaf_count, 0);
        assert_eq!(
            estimate.leaf_evaluation_backend,
            RelativeLeafEvaluationBackend::CpuBatch
        );
        assert_eq!(estimate.leaf_evaluation_batch_count, 3);
        assert_eq!(estimate.largest_leaf_evaluation_batch_size, 8);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: exact_depth_two_fixture_is_deterministic
    //
    // Description:
    //
    //   Verify exact depth-2 expansion has a stable leaf count and entropy value.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn exact_depth_two_fixture_is_deterministic() {

        // Run depth two twice to verify traversal order and partition math are stable.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator = RelativeFutureEntropyEstimator::default();
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let first_estimate = estimator.estimate(&taumax_position, &opponent_position, 2, &control);
        let second_estimate = estimator.estimate(&taumax_position, &opponent_position, 2, &control);

        // Startpos depth two expands twenty root moves and four hundred horizon leaves.

        assert_eq!(first_estimate.status, FutureEntropyStatus::Complete);
        assert_eq!(first_estimate.leaf_count, 400);
        assert_eq!(first_estimate.terminal_leaf_count, 0);
        assert_eq!(first_estimate.visited_node_count, 421);
        assert_close(first_estimate.value, second_estimate.value);
        assert_close(first_estimate.partition_sum, second_estimate.partition_sum);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: gpu_estimator_matches_cpu_estimator_when_available
    //
    // Description:
    //
    //   Verify the same future-estimator traversal matches CPU results when actual GPU execution is available.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn gpu_estimator_matches_cpu_estimator_when_available() {

        // Skip the parity check when the machine cannot initialize the optional GPU runtime.

        let gpu_kernel = match GpuRelativeLeafEvaluationKernel::try_new() {
            Ok(gpu_kernel) => gpu_kernel,
            Err(error) => {
                eprintln!("skipping GPU estimator parity check: {error}");
                return;
            }
        };
        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator =
            RelativeFutureEntropyEstimator::default().with_leaf_evaluation_batch_size(512);
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        // Evaluate the same tree with CPU and GPU leaf kernels.

        let cpu_estimate = estimator.estimate(&taumax_position, &opponent_position, 2, &control);
        let gpu_estimate = estimator.estimate_with_leaf_kernel(
            &taumax_position,
            &opponent_position,
            2,
            &control,
            gpu_kernel,
        );

        // Traversal counters must match exactly; floating-point partition values use f32 tolerance.

        assert_eq!(gpu_estimate.status, cpu_estimate.status);
        assert_eq!(gpu_estimate.leaf_count, cpu_estimate.leaf_count);
        assert_eq!(
            gpu_estimate.terminal_leaf_count,
            cpu_estimate.terminal_leaf_count
        );
        assert_eq!(
            gpu_estimate.visited_node_count,
            cpu_estimate.visited_node_count
        );
        assert_eq!(
            gpu_estimate.leaf_evaluation_backend,
            RelativeLeafEvaluationBackend::GpuBatch
        );
        assert_relative_close(
            gpu_estimate.partition_sum,
            cpu_estimate.partition_sum,
            1.0e-4,
        );
        assert_relative_close(gpu_estimate.value, cpu_estimate.value, 1.0e-4);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimator_uses_effective_taumax_depth
    //
    // Description:
    //
    //   Verify Max Depth and go-depth limits choose the requested exact horizon.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn estimator_uses_effective_taumax_depth() {

        // Set Max Depth higher than go depth so the command cap wins.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator = RelativeFutureEntropyEstimator::default();
        let mut configuration = EngineConfiguration::default();
        let mut limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        configuration.max_depth = 4;
        limits.depth = Some(1);

        let estimate = estimator.estimate_from_configuration(
            &taumax_position,
            &opponent_position,
            &configuration,
            &limits,
            &control,
        );

        // The requested estimate depth should match the effective capped depth.

        assert_eq!(estimate.requested_depth, 1);
        assert_eq!(estimate.leaf_count, 20);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimator_honors_cancellation
    //
    // Description:
    //
    //   Verify a stop request prevents tree expansion and returns a graceful fallback value.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn estimator_honors_cancellation() {

        // Request stop before estimation so recursion must return a fallback without visiting nodes.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator = RelativeFutureEntropyEstimator::default();
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        control.request_stop();

        let estimate = estimator.estimate(&taumax_position, &opponent_position, 2, &control);

        // Cancellation should still produce one finite non-terminal fallback leaf.

        assert_eq!(estimate.status, FutureEntropyStatus::Cancelled);
        assert_eq!(estimate.leaf_count, 1);
        assert_eq!(estimate.terminal_leaf_count, 0);
        assert_eq!(estimate.visited_node_count, 0);
        assert!(estimate.value.is_finite());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimator_degrades_when_node_budget_is_exceeded
    //
    // Description:
    //
    //   Verify budget overflow stops exact expansion and still returns a finite estimate.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn estimator_degrades_when_node_budget_is_exceeded() {

        // A budget of one lets the root visit occur, then forces fallback before deeper expansion.

        let (taumax_position, opponent_position) = paired_start_positions();
        let estimator = RelativeFutureEntropyEstimator::new(
            FreedomPotentialWeights::default(),
            DEFAULT_FUTURE_WEIGHT,
            1,
        );
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let estimate = estimator.estimate(&taumax_position, &opponent_position, 2, &control);

        // Budget degradation should remain finite and explicitly report the budget status.

        assert_eq!(estimate.status, FutureEntropyStatus::NodeBudgetExceeded);
        assert_eq!(estimate.leaf_count, 1);
        assert_eq!(estimate.terminal_leaf_count, 0);
        assert_eq!(estimate.visited_node_count, 1);
        assert!(estimate.value.is_finite());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: terminal_checkmate_scores_loss_boundary_for_side_to_move
    //
    // Description:
    //
    //   Verify a checkmated Taumax side receives the terminal loss boundary.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn terminal_checkmate_scores_loss_boundary_for_side_to_move() {

        // Black is checkmated in the Taumax position, so Black-as-Taumax receives the loss boundary.

        let taumax_position = Position::from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1").unwrap();
        let opponent_position = Position::startpos();
        let estimator = RelativeFutureEntropyEstimator::default();
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let estimate = estimator.estimate(&taumax_position, &opponent_position, 3, &control);

        // The terminal leaf contributes directly; no non-terminal leaf batching is needed.

        assert_eq!(estimate.status, FutureEntropyStatus::Complete);
        assert_eq!(estimate.leaf_count, 1);
        assert_eq!(estimate.terminal_leaf_count, 1);
        assert_eq!(estimate.visited_node_count, 1);
        assert_close(estimate.value, DEFAULT_TERMINAL_LOSS_SCORE);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: terminal_stalemate_scores_draw_boundary
    //
    // Description:
    //
    //   Verify stalemate receives the explicit draw boundary instead of immediate mobility potential.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn terminal_stalemate_scores_draw_boundary() {

        // Black is stalemated in the Taumax position, so the explicit draw boundary applies.

        let taumax_position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let opponent_position = Position::startpos();
        let estimator = RelativeFutureEntropyEstimator::default();
        let limits = SearchLimits::default();
        let control = SearchControl::from_limits(&limits);

        let estimate = estimator.estimate(&taumax_position, &opponent_position, 3, &control);

        // The entropy value is the log of exp(draw_score), which equals the draw score.

        assert_eq!(estimate.status, FutureEntropyStatus::Complete);
        assert_eq!(estimate.leaf_count, 1);
        assert_eq!(estimate.terminal_leaf_count, 1);
        assert_eq!(estimate.visited_node_count, 1);
        assert_close(estimate.value, DEFAULT_TERMINAL_DRAW_SCORE);
    }
}

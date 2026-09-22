//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Relative causal-entropy selector with adversarial opponent-reply scoring.
//
//----------------------------------------------------------------------------------------------------------------------

use std::time::Instant;

use cozy_chess::Move;

use crate::board::position::Position;
use crate::engine::configuration::{EngineConfiguration, EngineStrategy};
use crate::engine::horizon::effective_taumax_depth;
use crate::engine::relative::freedom::RelativeFreedomPotential;
use crate::engine::relative::future::{
    FutureEntropyEstimate, FutureEntropyStatus, RelativeFutureEntropyEstimator,
};
use crate::engine::relative::leaf::{
    GpuRelativeLeafEvaluationKernel, RelativeLeafEvaluationBackend,
};
use crate::engine::relative::profile::{
    relative_worker_count_for_root_moves, should_score_roots_in_parallel,
    RelativeAccelerationBackend, RelativeGpuAccelerationRequest, RelativeSearchProfile,
    RelativeSearchProfileCounters,
};
use crate::engine::relative::terminal::RelativeTerminalState;
use crate::engine::selector::{MoveSelection, MoveSelector};
use crate::engine::trace::RootMoveScore;
use crate::search::control::SearchControl;
use crate::search::limits::SearchLimits;

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeCausalEntropySelector
//
// Description:
//
//   Scores root moves by the opponent reply that minimizes Taumax future relative entropy.
//
//----------------------------------------------------------------------------------------------------------------------

pub struct RelativeCausalEntropySelector {
    future_entropy_estimator: RelativeFutureEntropyEstimator,
}

//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeLeafEvaluationRequest
//
// Description:
//
//   Selects the leaf-evaluation backend requested for one relative search.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum RelativeLeafEvaluationRequest {
    Cpu,
    Gpu(GpuRelativeLeafEvaluationKernel),
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationRequest
//
// Description:
//
//   Provides configuration conversion and estimator dispatch for leaf evaluation.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationRequest {

    //------------------------------------------------------------------------------------------------------------------
    // Function: from_configuration
    //
    // Description:
    //
    //   Create the requested leaf-evaluation backend for the current search.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn from_configuration(configuration: &EngineConfiguration) -> Self {

        // Convert the session configuration into the backend request used for this one search.

        if configuration.gpu {

            // Request GPU leaf evaluation, but keep CPU fallback inside the kernel.

            return Self::Gpu(GpuRelativeLeafEvaluationKernel::with_cpu_fallback());
        }

        // CPU is the universal backend and the default path.

        Self::Cpu
    }

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: gpu_acceleration_request
    //
    // Description:
    //
    //   Return whether the optional GPU backend was requested for trace diagnostics.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn gpu_acceleration_request(&self) -> RelativeGpuAccelerationRequest {

        // Report the user's GPU request separately from whether runtime GPU execution is available.

        match self {
            Self::Cpu => RelativeGpuAccelerationRequest::NotRequested,
            Self::Gpu(_) => RelativeGpuAccelerationRequest::Requested,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: estimate_future_entropy
    //
    // Description:
    //
    //   Estimate future entropy using the requested leaf-evaluation backend.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn estimate_future_entropy(
        &self,
        future_entropy_estimator: &RelativeFutureEntropyEstimator,
        taumax_position: &Position,
        opponent_position: &Position,
        depth: u64,
        control: &SearchControl,
    ) -> FutureEntropyEstimate {

        // Dispatch to the estimator with either the default CPU kernel or the requested GPU kernel.

        match self {
            Self::Cpu => future_entropy_estimator.estimate(
                taumax_position,
                opponent_position,
                depth,
                control,
            ),
            Self::Gpu(leaf_evaluation_kernel) => future_entropy_estimator
                .estimate_with_leaf_kernel(
                    taumax_position,
                    opponent_position,
                    depth,
                    control,
                    leaf_evaluation_kernel.clone(),
                ),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeCausalEntropySelector
//
// Description:
//
//   Provides construction and adversarial root-scoring behavior.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeCausalEntropySelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a relative causal-entropy selector with default estimator parameters.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new() -> Self {

        // Use the default estimator so normal construction has the standard weights and budgets.

        Self::with_future_entropy_estimator(RelativeFutureEntropyEstimator::default())
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: with_future_entropy_estimator
    //
    // Description:
    //
    //   Create a relative selector with a supplied future-entropy estimator.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_future_entropy_estimator(
        future_entropy_estimator: RelativeFutureEntropyEstimator,
    ) -> Self {

        // Store the estimator so tests can inject small budgets or alternate terminal policies.

        Self {
            future_entropy_estimator,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: score_root_move
    //
    // Description:
    //
    //   Score one root move by the opponent reply with the lowest resulting future entropy.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn score_root_move(
        &self,
        position: &Position,
        root_move: Move,
        depth: u64,
        control: &SearchControl,
        leaf_evaluation_request: &RelativeLeafEvaluationRequest,
    ) -> ScoredRootMove {

        // Record Taumax's original side so terminal states can be scored from the correct perspective.

        let taumax_side = position.board().side_to_move();
        let root_move_text = position.display_uci_move(root_move);
        let mut root_position = position.clone();

        // Apply the root move to get the board where the opponent replies.

        root_position
            .apply_move(root_move)
            .expect("generated legal root move must apply cleanly");

        // Generate opponent replies in deterministic order.

        let opponent_replies = sorted_legal_moves(&root_position);

        if opponent_replies.is_empty() {

            // A root move with no opponent replies is terminal or no-move; score it at the boundary.

            let relative_potential = RelativeFreedomPotential::measure(
                &root_position,
                &root_position,
                self.future_entropy_estimator.freedom_weights,
            );

            // Ask the terminal policy whether the no-reply position is a win, loss, or draw.

            let terminal_score = self
                .future_entropy_estimator
                .terminal_state_policy
                .score_position(&root_position, taumax_side);

            // Use the terminal boundary score when available; otherwise fall back to immediate freedom.

            let (score, terminal_state, terminal_leaf_count) = match terminal_score {
                Some((terminal_state, score)) => (score, Some(terminal_state), 1),
                None => (relative_potential.value, None, 0),
            };

            // Return a completed root score with no opponent-reply text.

            return ScoredRootMove {
                chess_move: root_move,
                move_text: root_move_text,
                score,
                depth,
                opponent_reply_text: None,
                status: FutureEntropyStatus::Complete,
                future_count: 1,
                own_legal_move_count: relative_potential
                    .taumax_freedom
                    .side_mobility
                    .legal_move_count,
                opponent_reply_count: 0,
                relative_potential_value: relative_potential.value,
                terminal_state,
                leaf_evaluation_backend: RelativeLeafEvaluationBackend::CpuBatch,
                profile_leaf_count: 1,
                terminal_leaf_count,
                leaf_evaluation_batch_count: 0,
                largest_leaf_evaluation_batch_size: 0,
                visited_node_count: 0,
            };
        }

        // Opponent reply consumes one ply, so future expansion receives the remaining depth.

        let future_depth = depth.saturating_sub(1);
        let mut worst_reply_score: Option<ScoredOpponentReply> = None;

        // Aggregate status and work counters across all opponent replies for this root move.

        let mut root_status = FutureEntropyStatus::Complete;
        let mut profile_leaf_count = 0;
        let mut terminal_leaf_count = 0;
        let mut leaf_evaluation_backend = RelativeLeafEvaluationBackend::CpuBatch;
        let mut leaf_evaluation_batch_count = 0;
        let mut largest_leaf_evaluation_batch_size = 0;
        let mut visited_node_count = 0;

        for opponent_reply in &opponent_replies {

            // Stop before starting another reply if the GUI or deadline requested cancellation.

            if control.should_stop() {
                return ScoredRootMove::cancelled(
                    root_move,
                    root_move_text,
                    depth,
                    opponent_replies.len(),
                );
            }

            // Apply this opponent reply to get the position Taumax must live with.

            let opponent_reply_text = root_position.display_uci_move(*opponent_reply);
            let mut reply_position = root_position.clone();

            reply_position
                .apply_move(*opponent_reply)
                .expect("generated legal opponent reply must apply cleanly");

            // Measure immediate relative potential after the reply for trace diagnostics.

            let relative_potential = RelativeFreedomPotential::measure(
                &reply_position,
                &root_position,
                self.future_entropy_estimator.freedom_weights,
            );

            // Estimate future relative entropy from the replied position.

            let future_estimate = leaf_evaluation_request.estimate_future_entropy(
                &self.future_entropy_estimator,
                &reply_position,
                &root_position,
                future_depth,
                control,
            );

            // Fold child estimator counters into this root's profiling totals.

            visited_node_count += future_estimate.visited_node_count;
            profile_leaf_count += future_estimate.leaf_count;
            terminal_leaf_count += future_estimate.terminal_leaf_count;
            leaf_evaluation_backend = future_estimate.leaf_evaluation_backend;
            leaf_evaluation_batch_count += future_estimate.leaf_evaluation_batch_count;
            largest_leaf_evaluation_batch_size = largest_leaf_evaluation_batch_size
                .max(future_estimate.largest_leaf_evaluation_batch_size);

            // Cancellation invalidates the partially scored root, so return a fallback placeholder.

            if future_estimate.status == FutureEntropyStatus::Cancelled {
                return ScoredRootMove::cancelled(
                    root_move,
                    root_move_text,
                    depth,
                    opponent_replies.len(),
                );
            }

            // Combine degradation status and keep the opponent reply with the lowest future score.

            root_status = root_status.combine(future_estimate.status);
            update_worst_reply_score(
                &mut worst_reply_score,
                opponent_reply_text,
                future_estimate,
                relative_potential,
            );
        }

        let worst_reply_score =
            worst_reply_score.expect("non-empty opponent reply list must produce a score");

        // The root move's value is its most restrictive opponent reply, modeling adversarial play.

        ScoredRootMove {
            chess_move: root_move,
            move_text: root_move_text,
            score: worst_reply_score.score,
            depth,
            opponent_reply_text: Some(worst_reply_score.move_text),
            status: root_status,
            future_count: worst_reply_score.future_count,
            own_legal_move_count: worst_reply_score.own_legal_move_count,
            opponent_reply_count: opponent_replies.len(),
            relative_potential_value: worst_reply_score.relative_potential_value,
            terminal_state: None,
            leaf_evaluation_backend,
            profile_leaf_count,
            terminal_leaf_count,
            leaf_evaluation_batch_count,
            largest_leaf_evaluation_batch_size,
            visited_node_count,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: score_root_moves_serial
    //
    // Description:
    //
    //   Score root moves on the current thread in deterministic order.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn score_root_moves_serial(
        &self,
        position: &Position,
        root_moves: &[Move],
        depth: u64,
        control: &SearchControl,
        leaf_evaluation_request: &RelativeLeafEvaluationRequest,
    ) -> Vec<ScoredRootMove> {

        // Score roots one at a time and preserve deterministic root order naturally.

        let mut scored_root_moves = Vec::new();

        for root_move in root_moves {

            // Stop between roots so cancellation still returns a legal fallback quickly.

            if control.should_stop() {
                break;
            }

            // Score this root move and remember whether it completed.

            let scored_root_move = self.score_root_move(
                position,
                *root_move,
                depth,
                control,
                leaf_evaluation_request,
            );
            let status = scored_root_move.status;

            scored_root_moves.push(scored_root_move);

            // Stop after a cancelled score so no partial diagnostics leak through.

            if status == FutureEntropyStatus::Cancelled {
                break;
            }
        }

        scored_root_moves
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: score_root_moves_parallel
    //
    // Description:
    //
    //   Score root moves with scoped worker threads while preserving root-order results.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn score_root_moves_parallel(
        &self,
        position: &Position,
        root_moves: &[Move],
        depth: u64,
        control: &SearchControl,
        worker_count: usize,
        leaf_evaluation_request: &RelativeLeafEvaluationRequest,
    ) -> Vec<ScoredRootMove> {

        // Split root moves into approximately equal chunks for scoped worker threads.

        let chunk_size = root_moves.len().div_ceil(worker_count);
        let mut indexed_scores: Vec<(usize, ScoredRootMove)> = Vec::new();

        std::thread::scope(|scope| {
            let mut handles = Vec::new();

            for (chunk_index, root_move_chunk) in root_moves.chunks(chunk_size).enumerate() {

                // Track the original root index so results can be sorted after workers join.

                let start_index = chunk_index * chunk_size;

                handles.push(scope.spawn(move || {
                    let mut chunk_scores = Vec::new();

                    for (offset, root_move) in root_move_chunk.iter().copied().enumerate() {

                        // Each worker also observes cancellation between root moves.

                        if control.should_stop() {
                            break;
                        }

                        // Score this root independently; all shared inputs are immutable.

                        let scored_root_move = self.score_root_move(
                            position,
                            root_move,
                            depth,
                            control,
                            leaf_evaluation_request,
                        );
                        let status = scored_root_move.status;

                        chunk_scores.push((start_index + offset, scored_root_move));

                        // Stop this chunk after cancellation to avoid returning partial scores.

                        if status == FutureEntropyStatus::Cancelled {
                            break;
                        }
                    }

                    chunk_scores
                }));
            }

            for handle in handles {

                // Join each worker and append its indexed scores into the aggregate list.

                let mut chunk_scores = handle
                    .join()
                    .expect("relative root scoring worker must not panic");

                indexed_scores.append(&mut chunk_scores);
            }
        });

        // Restore original root order so downstream tie-breaking is deterministic.

        indexed_scores.sort_by_key(|(root_move_index, _)| *root_move_index);

        indexed_scores
            .into_iter()
            .map(|(_, scored_root_move)| scored_root_move)
            .collect()
    }
}
//----------------------------------------------------------------------------------------------------------------------
// Implementation: Default for RelativeCausalEntropySelector
//
// Description:
//
//   Uses the normal constructor as the selector default.
//
//----------------------------------------------------------------------------------------------------------------------

impl Default for RelativeCausalEntropySelector {

    //------------------------------------------------------------------------------------------------------------------
    // Function: default
    //
    // Description:
    //
    //   Create the default selector.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn default() -> Self {

        // Use the standard constructor for Default.

        Self::new()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: MoveSelector for RelativeCausalEntropySelector
//
// Description:
//
//   Selects the root move whose most restrictive opponent reply leaves the best future entropy.
//
//----------------------------------------------------------------------------------------------------------------------

impl MoveSelector for RelativeCausalEntropySelector {

    //------------------------------------------------------------------------------------------------------------------
    // Method: select_move
    //
    // Description:
    //
    //   Score legal root moves and return the best fully scored move or a legal fallback.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn select_move(
        &mut self,
        position: &Position,
        limits: &SearchLimits,
        configuration: &EngineConfiguration,
        control: &SearchControl,
    ) -> MoveSelection {

        // Start profile timing before root filtering so diagnostics include all selector overhead.

        let profile_start_time = Instant::now();

        // Build the root candidate set, respecting any go searchmoves filter.

        let root_moves = filtered_sorted_root_moves(position, limits);
        let fallback_move = root_moves.first().copied();

        // Compute the effective search depth from engine configuration and go-command limits.

        let depth = effective_taumax_depth(configuration, limits);

        // Build the leaf backend request for this search.

        let leaf_evaluation_request =
            RelativeLeafEvaluationRequest::from_configuration(configuration);

        // Decide whether this root batch is large enough to score in parallel.

        let worker_count = relative_worker_count_for_root_moves(root_moves.len());
        let use_parallel = should_score_roots_in_parallel(root_moves.len(), worker_count)
            && !control.should_stop();
        let backend = if use_parallel {
            RelativeAccelerationBackend::ParallelCpu
        } else {
            RelativeAccelerationBackend::SerialCpu
        };

        // Score roots on the selected CPU path while preserving deterministic output order.

        let scored_root_moves = if use_parallel {
            self.score_root_moves_parallel(
                position,
                &root_moves,
                depth,
                control,
                worker_count,
                &leaf_evaluation_request,
            )
        } else {
            self.score_root_moves_serial(
                position,
                &root_moves,
                depth,
                control,
                &leaf_evaluation_request,
            )
        };

        // Aggregate completed root scores and profiling counters.

        let mut root_move_scores = Vec::new();
        let mut best_scored_root_move: Option<ScoredRootMove> = None;
        let mut profile_leaf_count = 0;
        let mut terminal_leaf_count = 0;
        let mut leaf_evaluation_backend = RelativeLeafEvaluationBackend::CpuBatch;
        let mut leaf_evaluation_batch_count = 0;
        let mut largest_leaf_evaluation_batch_size = 0;
        let mut visited_node_count = 0;

        for scored_root_move in scored_root_moves {

            // Do not emit cancelled partial scores.

            if scored_root_move.status == FutureEntropyStatus::Cancelled {
                break;
            }

            // Add this root's work counters into the search profile totals.

            profile_leaf_count += scored_root_move.profile_leaf_count;
            terminal_leaf_count += scored_root_move.terminal_leaf_count;
            leaf_evaluation_backend = scored_root_move.leaf_evaluation_backend;
            leaf_evaluation_batch_count += scored_root_move.leaf_evaluation_batch_count;
            largest_leaf_evaluation_batch_size = largest_leaf_evaluation_batch_size
                .max(scored_root_move.largest_leaf_evaluation_batch_size);
            visited_node_count += scored_root_move.visited_node_count;

            // Select the highest root score after each root has been adversarially minimized by reply.

            if best_scored_root_move
                .as_ref()
                .map(|best_root_move| scored_root_move.score > best_root_move.score)
                .unwrap_or(true)
            {
                best_scored_root_move = Some(scored_root_move.clone());
            }

            // Convert the root score into a protocol-safe trace record.

            root_move_scores.push(scored_root_move.to_root_move_score());
        }

        // Prefer the best completed score, falling back to the first legal root when scoring stopped early.

        let selected_move = best_scored_root_move
            .map(|scored_root_move| scored_root_move.chess_move)
            .or(fallback_move);

        // Build one profile line summarizing backend choice, work volume, and elapsed time.

        let profile = RelativeSearchProfile::from_counters(RelativeSearchProfileCounters {
            backend,
            leaf_evaluation_backend,
            gpu_acceleration_request: leaf_evaluation_request.gpu_acceleration_request(),
            root_move_count: root_moves.len(),
            scored_root_move_count: root_move_scores.len(),
            worker_count: if use_parallel { worker_count } else { 1 },
            visited_node_count,
            future_leaf_count: profile_leaf_count,
            terminal_leaf_count,
            leaf_evaluation_batch_count,
            largest_leaf_evaluation_batch_size,
            elapsed_duration: profile_start_time.elapsed(),
        });

        // Return the selected move, per-root traces, and the profile diagnostic line.

        MoveSelection::new(selected_move, root_move_scores)
            .with_diagnostic_line(profile.to_trace_line())
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: ScoredRootMove
//
// Description:
//
//   Stores the adversarial score for one legal root move.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct ScoredRootMove {
    chess_move: Move,
    move_text: String,
    score: f64,
    depth: u64,
    opponent_reply_text: Option<String>,
    status: FutureEntropyStatus,
    future_count: usize,
    own_legal_move_count: usize,
    opponent_reply_count: usize,
    relative_potential_value: f64,
    terminal_state: Option<RelativeTerminalState>,
    leaf_evaluation_backend: RelativeLeafEvaluationBackend,
    profile_leaf_count: usize,
    terminal_leaf_count: usize,
    leaf_evaluation_batch_count: usize,
    largest_leaf_evaluation_batch_size: usize,
    visited_node_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: ScoredRootMove
//
// Description:
//
//   Provides helper construction and trace conversion for root scores.
//
//----------------------------------------------------------------------------------------------------------------------

impl ScoredRootMove {

    //------------------------------------------------------------------------------------------------------------------
    // Function: cancelled
    //
    // Description:
    //
    //   Create a cancelled root score placeholder that is not emitted as a completed score.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn cancelled(
        chess_move: Move,
        move_text: String,
        depth: u64,
        opponent_reply_count: usize,
    ) -> Self {

        // Preserve the root move and reply count but zero the metrics so the caller will not emit it.

        Self {
            chess_move,
            move_text,
            score: 0.0,
            depth,
            opponent_reply_text: None,
            status: FutureEntropyStatus::Cancelled,
            future_count: 0,
            own_legal_move_count: 0,
            opponent_reply_count,
            relative_potential_value: 0.0,
            terminal_state: None,
            leaf_evaluation_backend: RelativeLeafEvaluationBackend::CpuBatch,
            profile_leaf_count: 0,
            terminal_leaf_count: 0,
            leaf_evaluation_batch_count: 0,
            largest_leaf_evaluation_batch_size: 0,
            visited_node_count: 0,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: to_root_move_score
    //
    // Description:
    //
    //   Convert the adversarial root score into the shared trace model.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn to_root_move_score(&self) -> RootMoveScore {

        // Start with common root-score fields and append relative-strategy diagnostics.

        let root_move_score = RootMoveScore::new(
            EngineStrategy::RelativeCausalEntropy,
            &self.move_text,
            self.score,
            self.depth,
        )
        .with_field(
            "opponent",
            self.opponent_reply_text.as_deref().unwrap_or("none"),
        )
        .with_field("own", self.own_legal_move_count.to_string())
        .with_field("opponentFreedom", self.opponent_reply_count.to_string())
        .with_field("rel", format!("{:.6}", self.relative_potential_value))
        .with_field("futures", self.future_count.to_string())
        .with_field("terminals", self.terminal_leaf_count.to_string());

        // Terminal roots get an extra boundary label; non-terminal roots omit the field.

        if let Some(terminal_state) = self.terminal_state {
            root_move_score.with_field("terminal", terminal_state.trace_value())
        } else {
            root_move_score
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: ScoredOpponentReply
//
// Description:
//
//   Stores a future-entropy score for one opponent reply.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct ScoredOpponentReply {
    move_text: String,
    score: f64,
    future_count: usize,
    own_legal_move_count: usize,
    relative_potential_value: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Function: update_worst_reply_score
//
// Description:
//
//   Keep the opponent reply that gives Taumax the lowest future entropy.
//
//----------------------------------------------------------------------------------------------------------------------

fn update_worst_reply_score(
    worst_reply_score: &mut Option<ScoredOpponentReply>,
    opponent_reply_text: String,
    future_estimate: FutureEntropyEstimate,
    relative_potential: RelativeFreedomPotential,
) {

    // Convert the reply estimate into the compact value compared for adversarial minimization.

    let scored_reply = ScoredOpponentReply {
        move_text: opponent_reply_text,
        score: future_estimate.value,
        future_count: future_estimate.leaf_count,
        own_legal_move_count: relative_potential
            .taumax_freedom
            .side_mobility
            .legal_move_count,
        relative_potential_value: relative_potential.value,
    };

    // Keep the reply with the lowest score because the opponent is assumed to restrict Taumax.

    if worst_reply_score
        .as_ref()
        .map(|worst_reply_score| scored_reply.score < worst_reply_score.score)
        .unwrap_or(true)
    {
        *worst_reply_score = Some(scored_reply);
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Function: filtered_sorted_root_moves
//
// Description:
//
//   Return legal root moves that satisfy searchmoves, sorted by UCI text.
//
//----------------------------------------------------------------------------------------------------------------------

fn filtered_sorted_root_moves(position: &Position, limits: &SearchLimits) -> Vec<Move> {

    // Start from sorted legal moves, then apply any GUI-supplied root filter.

    sorted_legal_moves(position)
        .into_iter()
        .filter(|chess_move| {
            let move_text = position.display_uci_move(*chess_move);

            limits.allows_root_move(&move_text)
        })
        .collect()
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

    // Sort by UCI text so serial and parallel scoring see a stable root order.

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

    //------------------------------------------------------------------------------------------------------------------
    // Function: trace_field_value
    //
    // Description:
    //
    //   Return one trace field value from a root-move score.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn trace_field_value(root_move_score: &RootMoveScore, name: &str) -> Option<String> {

        // Search by field name and clone the value so tests can compare owned strings.

        root_move_score
            .fields
            .iter()
            .find(|field| field.name == name)
            .map(|field| field.value.clone())
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: selected_move_text
    //
    // Description:
    //
    //   Return the selected move as UCI text.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn selected_move_text(position: &Position, selection: &MoveSelection) -> Option<String> {

        // Convert the optional selected move into the UCI text used by assertions.

        selection
            .selected_move
            .map(|chess_move| position.display_uci_move(chess_move))
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_selector_scores_root_moves
    //
    // Description:
    //
    //   Verify relative selection scores legal root moves instead of returning a placeholder.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn relative_selector_scores_root_moves() {

        // Use depth one so every legal startpos move receives a completed root score.

        let position = Position::startpos();
        let mut limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.depth = Some(1);
        configuration.max_depth = 1;

        let selection = selector.select_move(&position, &limits, &configuration, &control);

        // Verify all twenty startpos roots were scored and a profile line was attached.

        assert!(selection.selected_move.is_some());
        assert_eq!(selection.root_move_scores.len(), 20);
        assert!(selection.root_move_scores.iter().all(
            |root_move_score| root_move_score.strategy == EngineStrategy::RelativeCausalEntropy
        ));
        assert_eq!(selection.diagnostic_lines.len(), 1);
        assert!(selection.diagnostic_lines[0].starts_with("info string tau profile "));
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: root_score_records_restrictive_opponent_reply
    //
    // Description:
    //
    //   Verify a scored root move records the opponent reply that minimized its score.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn root_score_records_restrictive_opponent_reply() {

        // Restrict root candidates to e2e4 so the trace fields are easy to inspect.

        let position = Position::startpos();
        let mut limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.search_moves = vec!["e2e4".to_string()];
        limits.depth = Some(1);
        configuration.max_depth = 1;

        let selection = selector.select_move(&position, &limits, &configuration, &control);

        // Extract the relative diagnostic fields emitted for the single root score.

        let root_move_score = selection
            .root_move_scores
            .first()
            .expect("missing root score");
        let opponent_reply_text =
            trace_field_value(root_move_score, "opponent").expect("missing opponent reply");
        let own_legal_move_count = trace_field_value(root_move_score, "own")
            .expect("missing own move count")
            .parse::<usize>()
            .expect("invalid own move count");
        let relative_potential_value = trace_field_value(root_move_score, "rel")
            .expect("missing relative potential")
            .parse::<f64>()
            .expect("invalid relative potential");
        let future_count = trace_field_value(root_move_score, "futures")
            .expect("missing future count")
            .parse::<usize>()
            .expect("invalid future count");
        let terminal_count = trace_field_value(root_move_score, "terminals")
            .expect("missing terminal count")
            .parse::<usize>()
            .expect("invalid terminal count");

        // Verify the root score records the opponent reply and finite relative metrics.

        assert_eq!(selection.root_move_scores.len(), 1);
        assert_eq!(root_move_score.move_text, "e2e4");
        assert_ne!(opponent_reply_text, "none");
        assert!(own_legal_move_count > 0);
        assert_eq!(
            trace_field_value(root_move_score, "opponentFreedom"),
            Some("20".to_string())
        );
        assert!(relative_potential_value.is_finite());
        assert!(future_count > 0);
        assert_eq!(terminal_count, 0);
        assert_eq!(trace_field_value(root_move_score, "terminal"), None);
        assert_eq!(trace_field_value(root_move_score, "status"), None);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: searchmoves_limits_roots_but_not_opponent_replies
    //
    // Description:
    //
    //   Verify searchmoves filters only Taumax root moves and leaves opponent replies unrestricted.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn searchmoves_limits_roots_but_not_opponent_replies() {

        // Limit Taumax to e2e4 while allowing the opponent to choose from all legal replies.

        let position = Position::startpos();
        let mut limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.search_moves = vec!["e2e4".to_string()];
        limits.depth = Some(1);
        configuration.max_depth = 1;

        let selection = selector.select_move(&position, &limits, &configuration, &control);
        let opponent_reply_text =
            trace_field_value(&selection.root_move_scores[0], "opponent").unwrap();

        // The selected root is e2e4, but the opponent reply should be a different legal move.

        assert_eq!(
            selected_move_text(&position, &selection),
            Some("e2e4".to_string())
        );
        assert_ne!(opponent_reply_text, "e2e4");
        assert_ne!(opponent_reply_text, "none");
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: cancellation_returns_legal_fallback_without_partial_scores
    //
    // Description:
    //
    //   Verify cancellation before scoring returns the first legal root move allowed by searchmoves.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn cancellation_returns_legal_fallback_without_partial_scores() {

        // Request cancellation before selection so scoring should not emit completed root diagnostics.

        let position = Position::startpos();
        let mut limits = SearchLimits::default();
        let configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.search_moves = vec!["e2e4".to_string()];
        control.request_stop();

        let selection = selector.select_move(&position, &limits, &configuration, &control);

        // The fallback remains the first allowed legal root move.

        assert_eq!(
            selected_move_text(&position, &selection),
            Some("e2e4".to_string())
        );
        assert!(selection.root_move_scores.is_empty());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: mate_in_one_prefers_terminal_win_boundary
    //
    // Description:
    //
    //   Verify terminal wins outrank ordinary non-terminal freedom when both are root candidates.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn mate_in_one_prefers_terminal_win_boundary() {

        // Compare a quiet queen move with a checkmating queen move from the same position.

        let position = Position::from_fen("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1").unwrap();
        let mut limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.search_moves = vec!["g5e5".to_string(), "g5g7".to_string()];
        limits.depth = Some(1);
        configuration.max_depth = 1;

        let selection = selector.select_move(&position, &limits, &configuration, &control);
        let mate_score = selection
            .root_move_scores
            .iter()
            .find(|root_move_score| root_move_score.move_text == "g5g7")
            .expect("missing mate score");
        let quiet_score = selection
            .root_move_scores
            .iter()
            .find(|root_move_score| root_move_score.move_text == "g5e5")
            .expect("missing quiet score");

        // The terminal win should be selected and carry terminal trace fields.

        assert_eq!(
            selected_move_text(&position, &selection),
            Some("g5g7".to_string())
        );
        assert_eq!(
            trace_field_value(mate_score, "opponent"),
            Some("none".to_string())
        );
        assert_eq!(
            trace_field_value(mate_score, "terminal"),
            Some("win".to_string())
        );
        assert_eq!(
            trace_field_value(mate_score, "terminals"),
            Some("1".to_string())
        );
        assert!(mate_score.score > quiet_score.score);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: stalemate_root_is_terminal_draw_boundary
    //
    // Description:
    //
    //   Verify stalemate is traced as a draw terminal rather than a win.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn stalemate_root_is_terminal_draw_boundary() {

        // The forced root move creates stalemate, not checkmate.

        let position = Position::from_fen("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1").unwrap();
        let mut limits = SearchLimits::default();
        let mut configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        limits.search_moves = vec!["g5g6".to_string()];
        limits.depth = Some(1);
        configuration.max_depth = 1;

        let selection = selector.select_move(&position, &limits, &configuration, &control);
        let root_move_score = selection
            .root_move_scores
            .first()
            .expect("missing root score");

        // Verify the root was selected and labeled with the draw terminal boundary.

        assert_eq!(
            selected_move_text(&position, &selection),
            Some("g5g6".to_string())
        );
        assert_eq!(
            trace_field_value(root_move_score, "terminal"),
            Some("draw".to_string())
        );
        assert_eq!(
            trace_field_value(root_move_score, "terminals"),
            Some("1".to_string())
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: no_legal_root_move_returns_null_selection
    //
    // Description:
    //
    //   Verify positions with no legal root moves return no selected move.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn no_legal_root_move_returns_null_selection() {

        // This stalemate position has no legal root move for the side to move.

        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let limits = SearchLimits::default();
        let configuration = EngineConfiguration::default();
        let control = SearchControl::from_limits(&limits);
        let mut selector = RelativeCausalEntropySelector::new();

        let selection = selector.select_move(&position, &limits, &configuration, &control);

        // No move and no root diagnostics should be produced.

        assert_eq!(selection.selected_move, None);
        assert!(selection.root_move_scores.is_empty());
    }
}

// Numeric relative-freedom terms for one non-terminal future leaf.

struct RelativeLeafEvaluationFeatures {
    taumax_legal_move_term: f32,
    taumax_piece_mobility_term: f32,
    opponent_legal_move_term: f32,
    opponent_piece_mobility_term: f32,
};

// Scalar weights shared by every leaf in the dispatched batch.

struct RelativeLeafEvaluationParameters {
    piece_mobility_weight: f32,
    opponent_freedom_weight: f32,
    future_weight: f32,
    leaf_count: u32,
};

@group(0) @binding(0)
var<storage, read> leaf_features: array<RelativeLeafEvaluationFeatures>;

// One output contribution is written for each input leaf.

@group(0) @binding(1)
var<storage, read_write> partition_contributions: array<f32>;

@group(0) @binding(2)
var<uniform> parameters: RelativeLeafEvaluationParameters;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

    // Map each global invocation to one leaf index.

    let leaf_index = global_invocation_id.x;

    // Ignore extra invocations from the final partially filled workgroup.

    if (leaf_index >= parameters.leaf_count) {
        return;
    }

    // Compute Taumax freedom as legal-move entropy plus weighted per-piece mobility entropy.

    let features = leaf_features[leaf_index];
    let taumax_freedom = features.taumax_legal_move_term
        + parameters.piece_mobility_weight * features.taumax_piece_mobility_term;

    // Compute opponent freedom with the same piece-mobility weight.

    let opponent_freedom = features.opponent_legal_move_term
        + parameters.piece_mobility_weight * features.opponent_piece_mobility_term;

    // Compute relative potential as Taumax freedom minus weighted opponent freedom.

    let relative_potential = taumax_freedom
        - parameters.opponent_freedom_weight * opponent_freedom;

    // Compute the partition contribution exp(T * relative_potential) for this leaf.

    partition_contributions[leaf_index] = exp(parameters.future_weight * relative_potential);
}

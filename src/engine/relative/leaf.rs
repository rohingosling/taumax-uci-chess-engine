//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Batched relative-freedom leaf evaluation for CPU and optional GPU execution.
//
//----------------------------------------------------------------------------------------------------------------------

use std::sync::{mpsc, Arc};

use wgpu::util::DeviceExt;

use crate::board::position::Position;
use crate::engine::relative::freedom::{FreedomPotentialWeights, SideFreedom};

pub const DEFAULT_LEAF_EVALUATION_BATCH_SIZE: usize = 1024;

//----------------------------------------------------------------------------------------------------------------------
// Enum: RelativeLeafEvaluationBackend
//
// Description:
//
//   Identifies the backend used to evaluate non-terminal relative-freedom leaves.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativeLeafEvaluationBackend {
    CpuBatch,
    GpuBatch,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationBackend
//
// Description:
//
//   Provides trace formatting for leaf-evaluation backend labels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationBackend {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: trace_value
    //
    // Description:
    //
    //   Return the UCI-safe backend label used in profile traces.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn trace_value(self) -> &'static str {

        // Use the backend label that appears in profile trace fields.

        match self {
            Self::CpuBatch => "cpu-batch",
            Self::GpuBatch => "gpu-batch",
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeLeafEvaluationFeatures
//
// Description:
//
//   Stores the numeric feature vector used by one relative-freedom leaf evaluation.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativeLeafEvaluationFeatures {
    pub taumax_legal_move_term: f64,
    pub taumax_piece_mobility_term: f64,
    pub opponent_legal_move_term: f64,
    pub opponent_piece_mobility_term: f64,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationFeatures
//
// Description:
//
//   Provides feature extraction and scalar scoring for backend leaf kernels.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationFeatures {

    //------------------------------------------------------------------------------------------------------------------
    // Function: measure
    //
    // Description:
    //
    //   Extract the backend-neutral numeric features for one non-terminal relative-freedom leaf.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn measure(
        taumax_position: &Position,
        opponent_position: &Position,
        weights: FreedomPotentialWeights,
    ) -> Self {

        // Measure the immediate freedom terms for Taumax and opponent before discarding board state.

        let taumax_freedom = SideFreedom::measure(taumax_position, weights.piece_mobility_weight);
        let opponent_freedom =
            SideFreedom::measure(opponent_position, weights.piece_mobility_weight);

        // Store only scalar terms that CPU and GPU kernels can evaluate identically.

        Self {
            taumax_legal_move_term: taumax_freedom.legal_move_term,
            taumax_piece_mobility_term: taumax_freedom.piece_mobility_term,
            opponent_legal_move_term: opponent_freedom.legal_move_term,
            opponent_piece_mobility_term: opponent_freedom.piece_mobility_term,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: relative_potential_value
    //
    // Description:
    //
    //   Reconstruct the relative freedom-potential value from the backend-neutral feature vector.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn relative_potential_value(self, weights: FreedomPotentialWeights) -> f64 {

        // Compute Taumax freedom as legal-move entropy plus weighted per-piece mobility entropy.

        let taumax_freedom_value = self.taumax_legal_move_term
            + weights.piece_mobility_weight * self.taumax_piece_mobility_term;

        // Compute opponent freedom with the same piece-mobility weighting.

        let opponent_freedom_value = self.opponent_legal_move_term
            + weights.piece_mobility_weight * self.opponent_piece_mobility_term;

        // Compute relative potential as Taumax freedom minus weighted opponent freedom.

        taumax_freedom_value - weights.opponent_freedom_weight * opponent_freedom_value
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: partition_contribution
    //
    // Description:
    //
    //   Return this leaf's Boltzmann partition contribution for the requested future weighting.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn partition_contribution(
        self,
        weights: FreedomPotentialWeights,
        future_weight: f64,
    ) -> f64 {

        // Compute the Boltzmann-style partition contribution exp(T * relative_potential).

        (future_weight * self.relative_potential_value(weights)).exp()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeLeafEvaluationBatchResult
//
// Description:
//
//   Contains the result of evaluating one backend feature batch.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativeLeafEvaluationBatchResult {
    pub backend: RelativeLeafEvaluationBackend,
    pub partition_sum: f64,
    pub evaluated_leaf_count: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Trait: RelativeLeafEvaluationKernel
//
// Description:
//
//   Defines the backend contract for evaluating numeric relative-freedom leaf batches.
//
//----------------------------------------------------------------------------------------------------------------------

pub trait RelativeLeafEvaluationKernel: Clone {
    fn backend(&self) -> RelativeLeafEvaluationBackend;

    fn evaluate_batch(
        &self,
        leaf_features: &[RelativeLeafEvaluationFeatures],
        weights: FreedomPotentialWeights,
        future_weight: f64,
    ) -> RelativeLeafEvaluationBatchResult;
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: CpuRelativeLeafEvaluationKernel
//
// Description:
//
//   Evaluates relative-freedom feature batches on the CPU.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuRelativeLeafEvaluationKernel;

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationKernel for CpuRelativeLeafEvaluationKernel
//
// Description:
//
//   Provides the scalar CPU implementation of the leaf evaluation backend contract.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationKernel for CpuRelativeLeafEvaluationKernel {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: backend
    //
    // Description:
    //
    //   Return the trace backend used by this kernel.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn backend(&self) -> RelativeLeafEvaluationBackend {

        // The scalar implementation always reports CPU batch evaluation.

        RelativeLeafEvaluationBackend::CpuBatch
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: evaluate_batch
    //
    // Description:
    //
    //   Evaluate a feature batch with the same scalar math used by the direct relative potential.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn evaluate_batch(
        &self,
        leaf_features: &[RelativeLeafEvaluationFeatures],
        weights: FreedomPotentialWeights,
        future_weight: f64,
    ) -> RelativeLeafEvaluationBatchResult {

        // Compute the batch partition sum by summing every leaf's exponential contribution.

        RelativeLeafEvaluationBatchResult {
            backend: self.backend(),
            partition_sum: leaf_features
                .iter()
                .map(|features| features.partition_contribution(weights, future_weight))
                .sum(),
            evaluated_leaf_count: leaf_features.len(),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: GpuRelativeLeafEvaluationKernel
//
// Description:
//
//   Evaluates relative-freedom feature batches on an optional wgpu compute backend.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GpuRelativeLeafEvaluationKernel {
    runtime: Option<Arc<GpuLeafEvaluationRuntime>>,
    cpu_fallback_kernel: CpuRelativeLeafEvaluationKernel,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: GpuRelativeLeafEvaluationKernel
//
// Description:
//
//   Provides runtime GPU discovery and CPU fallback behavior.
//
//----------------------------------------------------------------------------------------------------------------------

impl GpuRelativeLeafEvaluationKernel {

    //------------------------------------------------------------------------------------------------------------------
    // Function: try_new
    //
    // Description:
    //
    //   Create a GPU leaf evaluator, or report why no compatible backend could be initialized.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn try_new() -> Result<Self, GpuLeafEvaluationError> {

        // Fail loudly when the caller explicitly requires a GPU runtime.

        Ok(Self {
            runtime: Some(Arc::new(GpuLeafEvaluationRuntime::try_new()?)),
            cpu_fallback_kernel: CpuRelativeLeafEvaluationKernel,
        })
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: with_cpu_fallback
    //
    // Description:
    //
    //   Create a GPU-requesting evaluator that silently falls back to CPU when initialization fails.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_cpu_fallback() -> Self {

        // Record a GPU runtime only if one initializes successfully; otherwise keep the CPU fallback.

        Self {
            runtime: GpuLeafEvaluationRuntime::try_new().ok().map(Arc::new),
            cpu_fallback_kernel: CpuRelativeLeafEvaluationKernel,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Predicate Accessor: is_available
    //
    // Description:
    //
    //   Return whether this kernel has an initialized GPU runtime.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn is_available(&self) -> bool {

        // Availability means this kernel currently owns an initialized runtime.

        self.runtime.is_some()
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationKernel for GpuRelativeLeafEvaluationKernel
//
// Description:
//
//   Evaluates batches on GPU when available and preserves CPU fallback semantics otherwise.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationKernel for GpuRelativeLeafEvaluationKernel {

    //------------------------------------------------------------------------------------------------------------------
    // Value Accessor: backend
    //
    // Description:
    //
    //   Return the requested backend; individual batch results report the backend that actually ran.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn backend(&self) -> RelativeLeafEvaluationBackend {

        // Report the backend that will be attempted for new batches.

        if self.runtime.is_some() {
            RelativeLeafEvaluationBackend::GpuBatch
        } else {
            RelativeLeafEvaluationBackend::CpuBatch
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: evaluate_batch
    //
    // Description:
    //
    //   Evaluate one batch on GPU when possible, otherwise use the scalar CPU fallback.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn evaluate_batch(
        &self,
        leaf_features: &[RelativeLeafEvaluationFeatures],
        weights: FreedomPotentialWeights,
        future_weight: f64,
    ) -> RelativeLeafEvaluationBatchResult {

        // Attempt GPU evaluation first, but treat any runtime error as a CPU fallback condition.

        if let Some(runtime) = &self.runtime {
            if let Ok(batch_result) = runtime.evaluate_batch(leaf_features, weights, future_weight)
            {
                return batch_result;
            }
        }

        // Preserve functional correctness on machines without compatible GPU support.

        self.cpu_fallback_kernel
            .evaluate_batch(leaf_features, weights, future_weight)
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: GpuLeafEvaluationError
//
// Description:
//
//   Stores a backend-neutral GPU initialization or execution failure message.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuLeafEvaluationError {
    message: String,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: GpuLeafEvaluationError
//
// Description:
//
//   Provides construction and display accessors for GPU backend failures.
//
//----------------------------------------------------------------------------------------------------------------------

impl GpuLeafEvaluationError {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a GPU error from displayable text.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn new(message: impl Into<String>) -> Self {

        // Store a plain string so the error stays backend-neutral at display time.

        Self {
            message: message.into(),
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Display for GpuLeafEvaluationError
//
// Description:
//
//   Formats GPU backend failures for diagnostics.
//
//----------------------------------------------------------------------------------------------------------------------

impl std::fmt::Display for GpuLeafEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        // Write only the stored message so callers can compose their own diagnostic prefix.

        formatter.write_str(&self.message)
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: Error for GpuLeafEvaluationError
//
// Description:
//
//   Marks GPU backend failures as standard errors.
//
//----------------------------------------------------------------------------------------------------------------------

impl std::error::Error for GpuLeafEvaluationError {}

//----------------------------------------------------------------------------------------------------------------------
// Struct: GpuLeafEvaluationRuntime
//
// Description:
//
//   Owns the wgpu device, queue, and compute pipeline for relative leaf evaluation.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Debug)]
struct GpuLeafEvaluationRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: GpuLeafEvaluationRuntime
//
// Description:
//
//   Provides wgpu setup and batch execution.
//
//----------------------------------------------------------------------------------------------------------------------

impl GpuLeafEvaluationRuntime {
    const WORKGROUP_SIZE: u32 = 64;

    //------------------------------------------------------------------------------------------------------------------
    // Function: try_new
    //
    // Description:
    //
    //   Initialize the first compatible high-performance wgpu adapter and compute pipeline.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn try_new() -> Result<Self, GpuLeafEvaluationError> {

        // Create a wgpu instance over the primary platform backends.

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::empty(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // Request a high-performance adapter because leaf batches are compute-heavy.

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GpuLeafEvaluationError::new("no compatible GPU adapter found"))?;

        // Request a device and queue with conservative downlevel limits for broad compatibility.

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("taumax-relative-leaf-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|error| {
            GpuLeafEvaluationError::new(format!("GPU device request failed: {error}"))
        })?;
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taumax-relative-leaf-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_leaf.wgsl").into()),
        });

        // Bind group layout: feature input buffer, contribution output buffer, and uniform parameters.

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taumax-relative-leaf-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Build the compute pipeline around the shader's main entry point.

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taumax-relative-leaf-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taumax-relative-leaf-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        // Store the initialized GPU objects for repeated batch execution.

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            compute_pipeline,
        })
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: evaluate_batch
    //
    // Description:
    //
    //   Execute the compute shader for one leaf feature batch and sum the readback contributions on CPU.
    //
    //------------------------------------------------------------------------------------------------------------------

    fn evaluate_batch(
        &self,
        leaf_features: &[RelativeLeafEvaluationFeatures],
        weights: FreedomPotentialWeights,
        future_weight: f64,
    ) -> Result<RelativeLeafEvaluationBatchResult, GpuLeafEvaluationError> {

        // Empty batches have a zero partition contribution and require no GPU work.

        if leaf_features.is_empty() {
            return Ok(RelativeLeafEvaluationBatchResult {
                backend: RelativeLeafEvaluationBackend::GpuBatch,
                partition_sum: 0.0,
                evaluated_leaf_count: 0,
            });
        }

        // Convert f64 CPU features into the f32 layout consumed by the shader.

        let gpu_leaf_features: Vec<GpuRelativeLeafEvaluationFeatures> = leaf_features
            .iter()
            .map(GpuRelativeLeafEvaluationFeatures::from)
            .collect();

        // Pack scalar weights and the active leaf count into one uniform parameter block.

        let parameters = GpuRelativeLeafEvaluationParameters {
            piece_mobility_weight: weights.piece_mobility_weight as f32,
            opponent_freedom_weight: weights.opponent_freedom_weight as f32,
            future_weight: future_weight as f32,
            leaf_count: gpu_leaf_features.len() as u32,
        };

        // Upload feature data as a read-only storage buffer.

        let features_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("taumax-relative-leaf-features"),
                contents: bytemuck::cast_slice(&gpu_leaf_features),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Upload weights and leaf count as a uniform buffer.

        let parameters_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("taumax-relative-leaf-parameters"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Allocate one f32 output contribution per input leaf.

        let output_byte_size = (gpu_leaf_features.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taumax-relative-leaf-output"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create a CPU-readable buffer used only for readback after the compute pass finishes.

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taumax-relative-leaf-readback"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind the buffers to the same binding indices declared in the WGSL shader.

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taumax-relative-leaf-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: features_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parameters_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("taumax-relative-leaf-encoder"),
            });

        {

            // Dispatch enough workgroups to cover every leaf, rounding up by the shader workgroup size.

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taumax-relative-leaf-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                parameters.leaf_count.div_ceil(Self::WORKGROUP_SIZE),
                1,
                1,
            );
        }

        // Copy shader output into the readback buffer before submitting the command encoder.

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_byte_size);
        self.queue.submit(Some(encoder.finish()));

        // Map readback asynchronously, then poll the device until the mapping completes.

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|error| {
                GpuLeafEvaluationError::new(format!("GPU readback channel failed: {error}"))
            })?
            .map_err(|error| {
                GpuLeafEvaluationError::new(format!("GPU readback mapping failed: {error}"))
            })?;

        // Compute the final partition sum on CPU by summing every f32 contribution returned by GPU.

        let mapped_range = buffer_slice.get_mapped_range();
        let partition_sum = bytemuck::cast_slice::<u8, f32>(&mapped_range)
            .iter()
            .map(|partition_contribution| *partition_contribution as f64)
            .sum();

        drop(mapped_range);
        readback_buffer.unmap();

        // Report the GPU backend and the number of leaves that were evaluated.

        Ok(RelativeLeafEvaluationBatchResult {
            backend: RelativeLeafEvaluationBackend::GpuBatch,
            partition_sum,
            evaluated_leaf_count: leaf_features.len(),
        })
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: GpuRelativeLeafEvaluationFeatures
//
// Description:
//
//   Provides the packed f32 feature layout consumed by the WGSL compute shader.
//
//----------------------------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRelativeLeafEvaluationFeatures {
    taumax_legal_move_term: f32,
    taumax_piece_mobility_term: f32,
    opponent_legal_move_term: f32,
    opponent_piece_mobility_term: f32,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: From RelativeLeafEvaluationFeatures for GpuRelativeLeafEvaluationFeatures
//
// Description:
//
//   Converts f64 CPU features into the f32 shader representation used by wgpu.
//
//----------------------------------------------------------------------------------------------------------------------

impl From<&RelativeLeafEvaluationFeatures> for GpuRelativeLeafEvaluationFeatures {
    fn from(features: &RelativeLeafEvaluationFeatures) -> Self {

        // Downcast each feature to f32 to match the shader's packed storage layout.

        Self {
            taumax_legal_move_term: features.taumax_legal_move_term as f32,
            taumax_piece_mobility_term: features.taumax_piece_mobility_term as f32,
            opponent_legal_move_term: features.opponent_legal_move_term as f32,
            opponent_piece_mobility_term: features.opponent_piece_mobility_term as f32,
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: GpuRelativeLeafEvaluationParameters
//
// Description:
//
//   Provides the packed uniform parameters consumed by the WGSL compute shader.
//
//----------------------------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRelativeLeafEvaluationParameters {
    piece_mobility_weight: f32,
    opponent_freedom_weight: f32,
    future_weight: f32,
    leaf_count: u32,
}

//----------------------------------------------------------------------------------------------------------------------
// Function: merge_leaf_evaluation_backends
//
// Description:
//
//   Preserve whether any evaluated batch actually used the GPU backend.
//
//----------------------------------------------------------------------------------------------------------------------

fn merge_leaf_evaluation_backends(
    current_backend: RelativeLeafEvaluationBackend,
    next_backend: RelativeLeafEvaluationBackend,
) -> RelativeLeafEvaluationBackend {

    // Once any batch used GPU, keep the aggregate backend labeled as GPU for profile diagnostics.

    if current_backend == RelativeLeafEvaluationBackend::GpuBatch
        || next_backend == RelativeLeafEvaluationBackend::GpuBatch
    {
        RelativeLeafEvaluationBackend::GpuBatch
    } else {
        RelativeLeafEvaluationBackend::CpuBatch
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeLeafEvaluationSummary
//
// Description:
//
//   Summarizes one completed set of batched leaf evaluations.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativeLeafEvaluationSummary {
    pub backend: RelativeLeafEvaluationBackend,
    pub partition_sum: f64,
    pub leaf_count: usize,
    pub batch_count: usize,
    pub largest_batch_size: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Struct: RelativeLeafEvaluationBatcher
//
// Description:
//
//   Collects non-terminal future leaves and evaluates them in deterministic CPU batches.
//
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RelativeLeafEvaluationBatcher<
    Kernel: RelativeLeafEvaluationKernel = CpuRelativeLeafEvaluationKernel,
> {
    freedom_weights: FreedomPotentialWeights,
    future_weight: f64,
    maximum_batch_size: usize,
    kernel: Kernel,
    effective_backend: RelativeLeafEvaluationBackend,
    pending_leaf_features: Vec<RelativeLeafEvaluationFeatures>,
    partition_sum: f64,
    leaf_count: usize,
    batch_count: usize,
    largest_batch_size: usize,
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationBatcher
//
// Description:
//
//   Provides CPU batch collection, flushing, and summary behavior.
//
//----------------------------------------------------------------------------------------------------------------------

impl RelativeLeafEvaluationBatcher<CpuRelativeLeafEvaluationKernel> {

    //------------------------------------------------------------------------------------------------------------------
    // Function: new
    //
    // Description:
    //
    //   Create a leaf evaluator with a bounded batch size.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn new(
        freedom_weights: FreedomPotentialWeights,
        future_weight: f64,
        maximum_batch_size: usize,
    ) -> Self {

        // Use the default CPU kernel for callers that only need scalar evaluation.

        Self::with_kernel(
            freedom_weights,
            future_weight,
            maximum_batch_size,
            CpuRelativeLeafEvaluationKernel,
        )
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Implementation: RelativeLeafEvaluationBatcher
//
// Description:
//
//   Provides backend-neutral batch collection, flushing, and summary behavior.
//
//----------------------------------------------------------------------------------------------------------------------

impl<Kernel: RelativeLeafEvaluationKernel> RelativeLeafEvaluationBatcher<Kernel> {

    //------------------------------------------------------------------------------------------------------------------
    // Function: with_kernel
    //
    // Description:
    //
    //   Create a leaf evaluator with an explicit backend kernel.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn with_kernel(
        freedom_weights: FreedomPotentialWeights,
        future_weight: f64,
        maximum_batch_size: usize,
        kernel: Kernel,
    ) -> Self {

        // Clamp the batch size to at least one so push_leaf can always make progress.

        Self {
            freedom_weights,
            future_weight,
            maximum_batch_size: maximum_batch_size.max(1),
            effective_backend: kernel.backend(),
            kernel,
            pending_leaf_features: Vec::new(),
            partition_sum: 0.0,
            leaf_count: 0,
            batch_count: 0,
            largest_batch_size: 0,
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: push_leaf
    //
    // Description:
    //
    //   Add one non-terminal fallback leaf to the current evaluation batch.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn push_leaf(&mut self, taumax_position: &Position, opponent_position: &Position) {

        // Convert the board pair into numeric features immediately and keep only those features queued.

        self.pending_leaf_features
            .push(RelativeLeafEvaluationFeatures::measure(
                taumax_position,
                opponent_position,
                self.freedom_weights,
            ));

        // Flush automatically when the pending batch reaches the configured maximum size.

        if self.pending_leaf_features.len() >= self.maximum_batch_size {
            self.flush();
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: flush
    //
    // Description:
    //
    //   Evaluate the pending leaves and add their Boltzmann weights to the partition sum.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn flush(&mut self) {

        // An empty flush is harmless and lets finish call this unconditionally.

        if self.pending_leaf_features.is_empty() {
            return;
        }

        // Evaluate the current batch and fold its partition contribution into the running totals.

        let current_batch_size = self.pending_leaf_features.len();
        let batch_result = self.kernel.evaluate_batch(
            &self.pending_leaf_features,
            self.freedom_weights,
            self.future_weight,
        );

        debug_assert_eq!(batch_result.evaluated_leaf_count, current_batch_size);

        // Track aggregate backend, leaf count, batch count, and largest observed batch size for profiling.

        self.effective_backend =
            merge_leaf_evaluation_backends(self.effective_backend, batch_result.backend);
        self.partition_sum += batch_result.partition_sum;
        self.leaf_count += current_batch_size;
        self.batch_count += 1;
        self.largest_batch_size = self.largest_batch_size.max(current_batch_size);
        self.pending_leaf_features.clear();
    }

    //------------------------------------------------------------------------------------------------------------------
    // Method: finish
    //
    // Description:
    //
    //   Flush all pending work and return a completed summary.
    //
    //------------------------------------------------------------------------------------------------------------------

    pub fn finish(&mut self) -> RelativeLeafEvaluationSummary {

        // Flush any partial batch before reporting a complete summary.

        self.flush();

        // Return a compact immutable summary for the future estimator and diagnostics.

        RelativeLeafEvaluationSummary {
            backend: self.effective_backend,
            partition_sum: self.partition_sum,
            leaf_count: self.leaf_count,
            batch_count: self.batch_count,
            largest_batch_size: self.largest_batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::relative::freedom::RelativeFreedomPotential;

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

        // Compare f64 values with a tight absolute tolerance for scalar formula checks.

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

        // Compute scale-normalized error so large partition sums can still be compared fairly.

        let scale = left_value.abs().max(right_value.abs()).max(1.0);
        let relative_difference = (left_value - right_value).abs() / scale;

        assert!(
            relative_difference <= tolerance,
            "left={left_value}, right={right_value}, relative={relative_difference}"
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: feature_vector_matches_direct_relative_potential
    //
    // Description:
    //
    //   Verify the backend-neutral feature vector preserves the relative freedom-potential formula.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn feature_vector_matches_direct_relative_potential() {

        // Measure both the direct potential and the backend feature vector from the same board pair.

        let position = Position::startpos();
        let weights = FreedomPotentialWeights::default();
        let direct_potential = RelativeFreedomPotential::measure(&position, &position, weights);
        let features = RelativeLeafEvaluationFeatures::measure(&position, &position, weights);

        // Verify feature reconstruction matches the scalar relative-potential formula.

        assert_close(
            features.relative_potential_value(weights),
            direct_potential.value,
        );

        // Verify the partition contribution is exp(relative potential) when future weight is one.

        assert_close(
            features.partition_contribution(weights, 1.0),
            direct_potential.value.exp(),
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: batcher_matches_direct_relative_potential_sum
    //
    // Description:
    //
    //   Verify CPU batch evaluation preserves the existing scalar leaf score.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn batcher_matches_direct_relative_potential_sum() {

        // Push one leaf so the batch sum should equal that leaf's direct exponential score.

        let position = Position::startpos();
        let weights = FreedomPotentialWeights::default();
        let direct_potential = RelativeFreedomPotential::measure(&position, &position, weights);
        let mut batcher = RelativeLeafEvaluationBatcher::new(weights, 1.0, 8);

        // Finish forces the pending single leaf to be evaluated.

        batcher.push_leaf(&position, &position);

        let summary = batcher.finish();

        assert_eq!(summary.backend, RelativeLeafEvaluationBackend::CpuBatch);
        assert_eq!(summary.leaf_count, 1);
        assert_eq!(summary.batch_count, 1);
        assert_eq!(summary.largest_batch_size, 1);
        assert_close(summary.partition_sum, direct_potential.value.exp());
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: batcher_flushes_by_configured_batch_size
    //
    // Description:
    //
    //   Verify batch counters expose the frontier shape used for backend profiling.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn batcher_flushes_by_configured_batch_size() {

        // A maximum batch size of two means three leaves should produce one full batch and one partial batch.

        let position = Position::startpos();
        let mut batcher =
            RelativeLeafEvaluationBatcher::new(FreedomPotentialWeights::default(), 1.0, 2);

        batcher.push_leaf(&position, &position);
        batcher.push_leaf(&position, &position);
        batcher.push_leaf(&position, &position);

        let summary = batcher.finish();

        assert_eq!(summary.leaf_count, 3);
        assert_eq!(summary.batch_count, 2);
        assert_eq!(summary.largest_batch_size, 2);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Struct: CountingLeafEvaluationKernel
    //
    // Description:
    //
    //   Provides a deterministic test kernel for proving the batcher uses its backend contract.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct CountingLeafEvaluationKernel;

    impl RelativeLeafEvaluationKernel for CountingLeafEvaluationKernel {
        fn backend(&self) -> RelativeLeafEvaluationBackend {

            // The counting test kernel is CPU-labeled because it never touches the GPU runtime.

            RelativeLeafEvaluationBackend::CpuBatch
        }

        fn evaluate_batch(
            &self,
            leaf_features: &[RelativeLeafEvaluationFeatures],
            _weights: FreedomPotentialWeights,
            _future_weight: f64,
        ) -> RelativeLeafEvaluationBatchResult {

            // Compute a deliberately simple partition sum: two points per leaf in the batch.

            RelativeLeafEvaluationBatchResult {
                backend: self.backend(),
                partition_sum: 2.0 * leaf_features.len() as f64,
                evaluated_leaf_count: leaf_features.len(),
            }
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: batcher_uses_supplied_leaf_kernel
    //
    // Description:
    //
    //   Verify alternate leaf kernels can be injected without changing estimator traversal code.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn batcher_uses_supplied_leaf_kernel() {

        // Inject the counting kernel so the expected partition sum is easy to prove.

        let position = Position::startpos();
        let mut batcher = RelativeLeafEvaluationBatcher::with_kernel(
            FreedomPotentialWeights::default(),
            1.0,
            2,
            CountingLeafEvaluationKernel,
        );

        batcher.push_leaf(&position, &position);
        batcher.push_leaf(&position, &position);
        batcher.push_leaf(&position, &position);

        let summary = batcher.finish();

        // Three leaves at two points each should sum to six.

        assert_eq!(summary.backend, RelativeLeafEvaluationBackend::CpuBatch);
        assert_eq!(summary.leaf_count, 3);
        assert_eq!(summary.batch_count, 2);
        assert_eq!(summary.largest_batch_size, 2);
        assert_close(summary.partition_sum, 6.0);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: gpu_kernel_with_cpu_fallback_returns_finite_batch_result
    //
    // Description:
    //
    //   Verify the optional GPU-requesting kernel can evaluate or fall back without affecting the CPU default path.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn gpu_kernel_with_cpu_fallback_returns_finite_batch_result() {

        // Use the fallback constructor so this test can pass on machines with or without a GPU.

        let position = Position::startpos();
        let mut batcher = RelativeLeafEvaluationBatcher::with_kernel(
            FreedomPotentialWeights::default(),
            1.0,
            8,
            GpuRelativeLeafEvaluationKernel::with_cpu_fallback(),
        );

        batcher.push_leaf(&position, &position);

        let summary = batcher.finish();

        // The backend may be CPU or GPU depending on runtime availability, but the result must be valid.

        assert!(matches!(
            summary.backend,
            RelativeLeafEvaluationBackend::CpuBatch | RelativeLeafEvaluationBackend::GpuBatch
        ));
        assert_eq!(summary.leaf_count, 1);
        assert_eq!(summary.batch_count, 1);
        assert_eq!(summary.largest_batch_size, 1);
        assert!(summary.partition_sum.is_finite());
        assert!(summary.partition_sum > 0.0);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: gpu_kernel_matches_cpu_partition_sum_when_available
    //
    // Description:
    //
    //   Verify actual GPU leaf-batch execution matches CPU semantics within f32 tolerance.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn gpu_kernel_matches_cpu_partition_sum_when_available() {

        // Skip only when no actual GPU runtime can be initialized.

        let gpu_kernel = match GpuRelativeLeafEvaluationKernel::try_new() {
            Ok(gpu_kernel) => gpu_kernel,
            Err(error) => {
                eprintln!("skipping GPU parity check: {error}");
                return;
            }
        };
        let weights = FreedomPotentialWeights::default();
        let mut taumax_position = Position::startpos();
        let mut opponent_position = Position::startpos();

        // Create a small set of distinct board-derived feature vectors.

        opponent_position.apply_uci_move("e2e4").unwrap();
        taumax_position.apply_uci_move("d2d4").unwrap();

        let base_features = [
            RelativeLeafEvaluationFeatures::measure(
                &Position::startpos(),
                &Position::startpos(),
                weights,
            ),
            RelativeLeafEvaluationFeatures::measure(&taumax_position, &opponent_position, weights),
            RelativeLeafEvaluationFeatures::measure(&opponent_position, &taumax_position, weights),
        ];

        // Repeat the base features to form a non-trivial GPU batch.

        let leaf_features = (0..256)
            .map(|index| base_features[index % base_features.len()])
            .collect::<Vec<_>>();

        // Evaluate the same batch on CPU and GPU for parity.

        let cpu_result =
            CpuRelativeLeafEvaluationKernel.evaluate_batch(&leaf_features, weights, 1.0);
        let gpu_result = gpu_kernel.evaluate_batch(&leaf_features, weights, 1.0);

        assert_eq!(cpu_result.backend, RelativeLeafEvaluationBackend::CpuBatch);
        assert_eq!(gpu_result.backend, RelativeLeafEvaluationBackend::GpuBatch);
        assert_eq!(
            gpu_result.evaluated_leaf_count,
            cpu_result.evaluated_leaf_count
        );
        assert_relative_close(gpu_result.partition_sum, cpu_result.partition_sum, 1.0e-4);
    }
}

use core::result::Result;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

#[cfg(feature = "benchmarks")]
pub mod eval;

pub mod definition;
mod kernel;

use crate::definition::{PoolError, PoolMode};
use crate::kernel::{
    backward::{pool2d_backward_launch_mode, pool2d_with_indices_backward_launch_mode},
    forward::{pool2d_launch_mode, pool2d_with_indices_launch_mode},
};

/// Pool2d public wrapper
///
/// Expects input in NHWC layout.
pub fn pool2d<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;

    pool2d_launch_mode(client, input, output, mode, dtype)
}

/// Pool2d with indices public wrapper
///
/// Expects input in NHWC layout. Output indices are expected to be in the same layout as well.
pub fn pool2d_with_indices<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    indices: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), output.shape.len())?;
    validate_rank(input.shape.len(), indices.shape.len())?;
    validate_nhwc_consistency(&input.shape, &output.shape)?;
    validate_nhwc_consistency(&input.shape, &indices.shape)?;

    pool2d_with_indices_launch_mode(client, input, output, indices, mode, dtype)
}

/// Pool2d backward public wrapper
///
/// Expects input and output gradients in NHWC layout.
pub fn pool2d_backward<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len())?;
    validate_rank(input.shape.len(), in_grad.shape.len())?;
    validate_nhwc_consistency(&input.shape, &out_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &in_grad.shape)?;

    pool2d_backward_launch_mode(client, input, out_grad, in_grad, mode, dtype)
}

/// Pool2d backward with indices public wrapper
///
/// Expects input and output gradients in NHWC layout. Output indices are expected to be in the same layout as well.
#[allow(clippy::too_many_arguments)]
pub fn pool2d_with_indices_backward<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    indices: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
    indices_dtype: StorageType,
) -> Result<(), PoolError> {
    validate_rank(input.shape.len(), out_grad.shape.len())?;
    validate_rank(input.shape.len(), in_grad.shape.len())?;
    validate_rank(input.shape.len(), indices.shape.len())?;
    validate_nhwc_consistency(&input.shape, &out_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &in_grad.shape)?;
    validate_nhwc_consistency(&input.shape, &indices.shape)?;

    pool2d_with_indices_backward_launch_mode(
        client,
        input,
        out_grad,
        indices,
        in_grad,
        mode,
        dtype,
        indices_dtype,
    )
}

/// Check that both tensors are 4D (Batch, Height, Width, Channels).
fn validate_rank(input_rank: usize, output_rank: usize) -> Result<(), PoolError> {
    if input_rank != 4 || output_rank != 4 {
        return Err(PoolError::InvalidRank {
            input: input_rank,
            output: output_rank,
        });
    }
    Ok(())
}

/// Check that Batch (0) and Channel (3) dimensions match.
/// Height (1) and Width (2) are allowed to differ for resizing.
fn validate_nhwc_consistency(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), PoolError> {
    if input_shape[0] != output_shape[0] {
        return Err(PoolError::BatchMismatch {
            input: input_shape[0],
            output: output_shape[0],
        });
    }

    if input_shape[3] != output_shape[3] {
        return Err(PoolError::ChannelMismatch {
            input: input_shape[3],
            output: output_shape[3],
        });
    }

    Ok(())
}

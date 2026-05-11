mod adaptive_avg_pool2d_backward;
mod avg_pool2d_backward;
mod max_pool2d_backward;

pub(crate) use adaptive_avg_pool2d_backward::*;
pub(crate) use avg_pool2d_backward::*;
pub(crate) use max_pool2d_backward::*;

use crate::definition::{PoolError, PoolMode};
use cubecl::prelude::*;
use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

pub(crate) fn pool2d_backward_launch_mode<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    match mode {
        PoolMode::Avg(avg_options) => {
            avg_pool2d_backward_launch(client, input, out_grad, in_grad, avg_options, dtype)
        }
        PoolMode::AdaptiveAvg(adaptive_avg_options) => adaptive_avg_pool2d_backward_launch(
            client,
            input,
            out_grad,
            in_grad,
            adaptive_avg_options,
            dtype,
        ),
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pool2d_with_indices_backward_launch_mode<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    indices: TensorBinding<R>,
    in_grad: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
    indices_dtype: StorageType,
) -> Result<(), PoolError> {
    match mode {
        PoolMode::Max(max_options) => max_pool2d_with_indices_backward_launch(
            client,
            input,
            out_grad,
            indices,
            in_grad,
            max_options,
            dtype,
            indices_dtype,
        ),
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

#[cube]
pub(crate) fn loop_ranges(
    ih: i32,
    iw: i32,
    grad_h: u32,
    grad_w: u32,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
) -> (u32, u32, u32, u32) {
    let kms_0 = args.dilation_0 * kernel_size_0 - args.stride_0;
    let kms_1 = args.dilation_1 * kernel_size_1 - args.stride_1;

    let oh_start = clamp_min((ih + args.padding_0 - kms_0) / args.stride_0, 0) as u32;
    let ow_start = clamp_min((iw + args.padding_1 - kms_1) / args.stride_1, 0) as u32;
    let oh_end = clamp_max(clamp_min(kms_0, 0) as u32 + oh_start, grad_h - 1) + 1;
    let ow_end = clamp_max(clamp_min(kms_1, 0) as u32 + ow_start, grad_w - 1) + 1;

    (oh_start, oh_end, ow_start, ow_end)
}

mod adaptive_avg_pool2d;
mod avg_pool2d;
mod max_pool2d;
mod pool;

pub(crate) use adaptive_avg_pool2d::*;
pub(crate) use avg_pool2d::*;
pub(crate) use max_pool2d::*;
pub(crate) use pool::*;

use crate::definition::{PoolError, PoolMode};
use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding, prelude::*};

pub(crate) fn pool2d_launch_mode<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    match mode {
        PoolMode::Max(max_options) => max_pool2d_launch(client, input, output, max_options, dtype),
        PoolMode::Avg(avg_options) => avg_pool2d_launch(client, input, output, avg_options, dtype),
        PoolMode::AdaptiveAvg(adaptive_avg_options) => {
            adaptive_avg_pool2d_launch(client, input, output, adaptive_avg_options, dtype)
        }
    }
}

pub(crate) fn pool2d_with_indices_launch_mode<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    indices: TensorBinding<R>,
    mode: PoolMode<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    match mode {
        PoolMode::Max(max_options) => {
            max_pool2d_with_indices_launch(client, input, output, indices, max_options, dtype)
        }
        _ => Err(PoolError::UnsupportedMode {
            mode: format!("{0:?}", mode),
        }),
    }
}

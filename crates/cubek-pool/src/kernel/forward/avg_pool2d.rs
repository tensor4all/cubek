use super::{
    super::{address_type_for, launch_config_for, shape_divmod},
    pool2d::{Pool2dArgsLaunch, Pool2dStrategy, Pool2dStrategyFamily, Position, pool2d, view4d},
};
use crate::definition::{AvgPoolOptions, PoolError};
use cubecl::{Runtime, num_traits::Zero, prelude::TensorBinding, prelude::*, std::tensor::View};

struct AvgPoolStrategy;

impl Pool2dStrategyFamily for AvgPoolStrategy {
    type Indices<N: Size> = ();
    type Config = AvgPoolStrategyConfig;
    type Pool2d<T: Numeric, N: Size> = Self;
}

#[derive(CubeType, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct AvgPoolStrategyConfig {
    count_include_pad: bool,
    /// Total padded height (input_height + 2 * padding_0)
    padded_h: u32,
    /// Total padded width (input_width + 2 * padding_1)
    padded_w: u32,
}

#[cube]
impl<T: Numeric, N: Size> Pool2dStrategy<T, N> for AvgPoolStrategy {
    type Accumulator = (Vector<T, N>, u32);
    type Config = AvgPoolStrategyConfig;
    type Indices = ();

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        let sum = Vector::zero();
        // Count will be set dynamically: either by accumulate (count_include_pad=false)
        // or by set_padded_count (count_include_pad=true)
        let count = 0u32;

        (sum, count)
    }

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: usize,
        result: Vector<T, N>,
    ) {
        let (sum, count) = accumulator;

        // Only count valid positions when count_include_pad=false
        if comptime![!config.count_include_pad] {
            *count += 1;
        }

        *sum += result;
    }

    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        ih: u32,
        iw: u32,
    ) {
        // When count_include_pad=true, count positions within padded bounds
        // (excludes ceil_mode extensions beyond the padded input)
        if comptime![config.count_include_pad] && ih < config.padded_h && iw < config.padded_w {
            let (_sum, count) = accumulator;
            *count += 1;
        }
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        let (sum, count) = accumulator;
        output[position] = sum / Vector::cast_from(count);
    }
}

pub(crate) fn avg_pool2d_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: AvgPoolOptions<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    let [_, in_h, in_w, _] = input.shape.dims();
    let dilation = 1;

    let launch = launch_config_for(client, dtype, &input, &output);
    let address_type = address_type_for((&input, dtype.size()), &[(&output, dtype.size())]);

    let padded_0 = in_h as u32 + 2u32 * options.window.padding[0] as u32;
    let padded_1 = in_w as u32 + 2u32 * options.window.padding[1] as u32;

    pool2d::launch::<AvgPoolStrategy, R>(
        client,
        launch.cube_count,
        launch.cube_dim,
        address_type,
        launch.vector_size,
        input.into_tensor_arg(),
        view4d(output.clone(), launch.vector_size),
        (),
        shape_divmod(&output),
        launch.working_units,
        Pool2dArgsLaunch::new(
            options.window.stride[0] as u32,
            options.window.stride[1] as u32,
            dilation as u32,
            dilation as u32,
            options.window.padding[0] as u32,
            options.window.padding[1] as u32,
        ),
        (
            options.window.kernel_size[0] as u32,
            options.window.kernel_size[1] as u32,
        ),
        AvgPoolStrategyConfig {
            count_include_pad: options.count_include_pad,
            padded_h: padded_0,
            padded_w: padded_1,
        },
        dtype,
    );

    Ok(())
}

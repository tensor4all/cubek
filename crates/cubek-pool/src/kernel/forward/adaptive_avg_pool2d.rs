use super::{
    super::{
        address_type_for, decompose_linear, end_index, launch_config_for, shape_divmod, start_index,
    },
    pool2d::{Position, view4d},
};
use crate::definition::{AdaptiveAvgPoolOptions, PoolError};
use cubecl::{
    Runtime,
    num_traits::Zero,
    prelude::TensorBinding,
    prelude::*,
    std::{FastDivmod, tensor::View},
};

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool2d_direct<E: Numeric, N: Size>(
    input: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (b, oh, ow, c) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);

    let (_, out_h, out_w, _) = output.shape();
    let (in_stride_h, in_stride_w) = (input.stride(1), input.stride(2));
    let (in_h, in_w) = (input.shape(1), input.shape(2));

    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);

    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Vector::zero();

    let index_input_base = b * input.stride(0) + c * input.stride(3);

    for ih in ih_start..ih_end {
        let index_input_2 = ih * in_stride_h;

        for iw in iw_start..iw_end {
            let index_input_3 = iw * in_stride_w;

            let index_input = index_input_base + index_input_2 + index_input_3;
            sum += input[index_input / input.vector_size()];
        }
    }

    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[(b, oh, ow, c)] = sum / Vector::cast_from(num_ih * num_iw);
}

pub(crate) fn adaptive_avg_pool2d_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    _options: AdaptiveAvgPoolOptions<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    let launch = launch_config_for(client, dtype, &input, &output);
    let address_type = address_type_for((&input, dtype.size()), &[(&output, dtype.size())]);

    adaptive_avg_pool2d_direct::launch(
        client,
        launch.cube_count,
        launch.cube_dim,
        address_type,
        launch.vector_size,
        input.into_tensor_arg(),
        view4d(output.clone(), launch.vector_size),
        shape_divmod(&output),
        launch.working_units,
        dtype,
    );

    Ok(())
}

pub(crate) mod backward;
pub(crate) mod forward;

use cubecl::{
    CubeCount, CubeDim, Runtime, calculate_cube_count_elemwise, prelude::*, std::FastDivmod,
    tensor_vector_size_parallel,
};

pub(crate) struct LaunchConfig {
    pub vector_size: VectorSize,
    pub working_units: usize,
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

pub(crate) fn launch_config_for<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: StorageType,
    input: &TensorBinding<R>,
    output: &TensorBinding<R>,
) -> LaunchConfig {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    LaunchConfig {
        vector_size,
        working_units,
        cube_dim,
        cube_count,
    }
}

pub(crate) fn address_type_for<R: Runtime>(
    first: (&TensorBinding<R>, usize),
    rest: &[(&TensorBinding<R>, usize)],
) -> AddressType {
    let mut address_type = first.0.required_address_type(first.1);
    for (binding, dtype_size) in rest {
        address_type = address_type.max(binding.required_address_type(*dtype_size));
    }
    address_type
}

pub(crate) fn shape_divmod<R: Runtime>(
    binding: &TensorBinding<R>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

#[cube]
pub(crate) fn decompose_linear(
    index: usize,
    shape: &Sequence<FastDivmod<usize>>,
) -> (usize, usize, usize, usize) {
    let (remainder, c) = shape[3].div_mod(index);
    let (remainder, ow) = shape[2].div_mod(remainder);
    let (remainder, oh) = shape[1].div_mod(remainder);
    let (_, b) = shape[0].div_mod(remainder);

    (b, oh, ow, c)
}

#[cube]
pub(crate) fn start_index(
    output_size_index: usize,
    output_size: usize,
    input_size: usize,
) -> usize {
    (output_size_index * input_size) / output_size
}

#[cube]
pub(crate) fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}

//! Isolated tests for the shared-memory loading logic.
//!
//! The forward kernel couples loading the input tile into shared memory with
//! the subsequent weighted reads, which makes the loading hard to reason about
//! on its own. This test runs the *real* loading path — it fills an actual
//! `Shared` tile with [`load_shared_region`] (the same code production uses,
//! via [`SharedMemoryReader::new`]) — and then, purely as an observation step
//! outside the interpolate flow, has a single unit copy the whole tile verbatim
//! into the output. That lets us check "which input element lands in which slot"
//! against a tiny CPU reference, with no interpolation maths involved.

use cubecl::{TestRuntime, prelude::*, tensor_vector_size_parallel};
use cubek_interpolate::{components::readers::load_shared_region, routines::SharedMemoryBlueprint};
use cubek_test_utils::{TestInput, assert_equals_approx};

use super::{build_output_tensor, output_host_f32};

/// Fills a real `Shared` tile via the production loading path, then dumps it.
///
/// All units cooperatively load the region into shared memory (exactly as the
/// forward kernel does), sync, and then a single unit copies every slot into
/// `output`. `output` is laid out exactly like the shared-memory tile — NHWC
/// `[1, smem_height, smem_width, channel_groups * vector_size]` — so slot `i`
/// (in vector units) maps to the same flat index in both.
#[cube(launch_unchecked)]
fn load_region_kernel<N: Size>(
    input: &Tensor<Vector<f32, N>>,
    output: &mut Tensor<Vector<f32, N>>,
    #[comptime] batch: usize,
    #[comptime] min_row: isize,
    #[comptime] min_col: isize,
    #[comptime] blueprint: SharedMemoryBlueprint,
) {
    let input_height = input.shape(1);
    let input_width = input.shape(2);
    let vector_size = N::value();

    let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;
    let mut smem = Shared::new_slice(smem_size);

    load_shared_region::<f32, f32, N, Shared<[Vector<f32, N>]>>(
        input,
        &mut smem,
        batch,
        input_height,
        input_width,
        min_row,
        min_col,
        vector_size,
        blueprint,
    );

    sync_cube();

    // Apart from the real interpolate flow: a single unit writes the raw tile
    // contents into the output so the test sees exactly what was loaded.
    if UNIT_POS == 0 {
        for i in 0..smem_size {
            output[i] = smem[i];
        }
    }
}

/// Run the loading kernel for one region and assert it matches a host gather.
///
/// `[smem_height, smem_width]` is the region size; `(min_row, min_col)` its
/// top-left corner in the (possibly negative / out-of-bounds) input space. The
/// number of `channel_groups` is derived from the channel count and the chosen
/// vector size so the whole channel axis is covered.
fn run_load_region_test(
    input_shape: [usize; 4],
    batch: usize,
    min_row: isize,
    min_col: isize,
    smem_height: usize,
    smem_width: usize,
) {
    let client = TestRuntime::client(&Default::default());
    let [_, input_height, input_width, channels] = input_shape;

    let (input, input_data) = TestInput::builder(client.clone(), input_shape.to_vec())
        .uniform(1234, -1.0, 1.0)
        .generate_with_f32_host_data();

    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(input.dtype.size()),
        input.shape(),
        input.strides(),
        input.shape().len() - 1,
    );
    let vector_size = vector_size as usize;
    assert_eq!(
        channels % vector_size,
        0,
        "channels ({channels}) must be divisible by the vector size ({vector_size})"
    );
    let channel_groups = channels / vector_size;

    let blueprint = SharedMemoryBlueprint {
        smem_width,
        smem_height,
        channel_groups,
    };

    // The output mirrors the shared-memory tile, one element per slot.
    let output_shape = vec![1, smem_height, smem_width, channels];
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);

    let smem_size = smem_width * smem_height * channel_groups;
    let cube_dim = CubeDim::new(&client, smem_size);

    unsafe {
        load_region_kernel::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            vector_size,
            input.clone().binding().into_tensor_arg(),
            output.clone().binding().into_tensor_arg(),
            batch,
            min_row,
            min_col,
            blueprint,
        );
    }

    let expected = host_reference(
        &input_data,
        batch,
        input_height,
        input_width,
        min_row,
        min_col,
        smem_height,
        smem_width,
        channels,
    );
    let expected = TestInput::builder(client.clone(), output_shape)
        .custom(expected)
        .generate_with_f32_host_data()
        .1;

    let actual = output_host_f32(&client, output);
    assert_equals_approx(&actual, &expected, 0.0)
        .as_test_outcome()
        .enforce();
}

/// CPU reference for the gather, in row-major order of the output tile.
/// Out-of-bounds neighbours are clamped to the input edge (NHWC layout).
#[allow(clippy::too_many_arguments)]
fn host_reference(
    input: &cubek_test_utils::HostData,
    batch: usize,
    input_height: usize,
    input_width: usize,
    min_row: isize,
    min_col: isize,
    smem_height: usize,
    smem_width: usize,
    channels: usize,
) -> Vec<f32> {
    let clamp = |v: isize, max: usize| v.max(0).min(max as isize - 1) as usize;

    let mut data = Vec::with_capacity(smem_height * smem_width * channels);
    for local_row in 0..smem_height {
        for local_col in 0..smem_width {
            let global_row = clamp(min_row + local_row as isize, input_height);
            let global_col = clamp(min_col + local_col as isize, input_width);
            for channel in 0..channels {
                data.push(input.get_f32(&[batch, global_row, global_col, channel]));
            }
        }
    }
    data
}

#[test]
fn load_region_interior_single_channel() {
    // Region fully inside the input, no clamping, scalar channels.
    run_load_region_test([1, 8, 8, 1], 0, 2, 3, 3, 4);
}

#[test]
fn load_region_clamp_top_left() {
    // Top-left corner is out of bounds: rows/cols below 0 clamp to the edge.
    run_load_region_test([1, 6, 6, 4], 0, -2, -1, 4, 4);
}

#[test]
fn load_region_clamp_bottom_right() {
    // Region runs past the bottom-right edge and must clamp there.
    run_load_region_test([1, 5, 5, 4], 0, 3, 4, 4, 4);
}

#[test]
fn load_region_multiple_channel_groups() {
    // 8 channels with a vector size > 1 exercises channel-group striding.
    run_load_region_test([1, 6, 7, 8], 0, 1, 1, 3, 3);
}

#[test]
fn load_region_non_zero_batch() {
    // A batch offset must be applied to the input base.
    run_load_region_test([3, 6, 6, 4], 2, 0, 0, 4, 4);
}

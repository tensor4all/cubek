use cubecl::prelude::*;
use cubecl::std::tensor::{AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand};

use crate::{
    ComplexTensorBinding, ComplexTensorHandle, FftError, FftNormalization,
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
    },
    interleaved_layout::InterleavedBatchSignalLayout,
};

/// Runs a C32 FFT over an interleaved complex tensor along `dim`.
pub fn cfft_interleaved<R: Runtime>(
    input: ComplexTensorHandle<R>,
    dim: usize,
    mode: FftMode,
    normalization: FftNormalization,
) -> Result<ComplexTensorHandle<R>, FftError> {
    let shape = input.shape().to_vec();
    let client = R::client(&Default::default());
    let plan = cfft_plan(&client, &shape, dim)?;
    normalization.scale_f32(plan.n_fft)?;

    let strides = input.strides().to_vec();
    ensure_non_overlapping_output_layout(&shape, &strides)?;
    let dtype = input.dtype();
    let byte_len = input
        .physical_scalar_len()
        .checked_mul(dtype.size())
        .ok_or(FftError::SizeOverflow)?;
    let output = ComplexTensorHandle::new_strided(shape, strides, client.empty(byte_len), dtype)?;

    cfft_interleaved_launch(
        &client,
        input.binding(),
        output.binding(),
        dim,
        mode,
        normalization,
    )?;
    Ok(output)
}

/// Launches a shared-memory C32 FFT into an interleaved output tensor.
pub fn cfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: ComplexTensorBinding<'_, R>,
    output: ComplexTensorBinding<'_, R>,
    dim: usize,
    mode: FftMode,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let input_shape = input.shape();
    let plan = cfft_plan(client, input_shape, dim)?;
    if output.shape() != input_shape {
        return Err(FftError::ShapeMismatch {
            name: "output",
            actual: output.shape().to_vec(),
            expected: input_shape.to_vec(),
        });
    }
    if output.dtype() != input.dtype() {
        return Err(FftError::UnsupportedDtype {
            actual: output.dtype(),
        });
    }
    ensure_non_overlapping_output_layout(output.shape(), output.strides())?;

    normalization.scale_f32(plan.n_fft)?;

    output.ensure_unique_output()?;
    if plan.count == 0 {
        return Ok(());
    }

    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_cube = (plan.n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, plan.count, CubeDim::new_single());
    let input_tensor = input.tensor();
    let output_tensor = output.tensor();

    cfft_interleaved_shared_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        input_tensor.into_tensor_arg(),
        output_tensor.into_tensor_arg(),
        plan.count_u32,
        plan.n_fft,
        log2_n,
        threads_per_cube,
        dim,
        mode,
        normalization,
    );
    Ok(())
}

struct CfftPlan {
    n_fft: usize,
    count: usize,
    count_u32: u32,
}

fn cfft_plan<R: Runtime>(
    client: &ComputeClient<R>,
    shape: &[usize],
    dim: usize,
) -> Result<CfftPlan, FftError> {
    validate_fft_shape(shape, dim)?;
    let n_fft = shape[dim];
    let max_n = max_shared_fft_n(client);
    if n_fft > max_n {
        return Err(FftError::InvalidLength {
            name: "n_fft",
            value: n_fft,
            min: 2,
            max: max_n,
        });
    }
    let count = shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != dim)
        .try_fold(1usize, |count, (_, extent)| {
            count.checked_mul(*extent).ok_or(FftError::SizeOverflow)
        })?;
    let count_u32 = u32::try_from(count).map_err(|_| FftError::SizeOverflow)?;
    Ok(CfftPlan {
        n_fft,
        count,
        count_u32,
    })
}

fn ensure_non_overlapping_output_layout(
    shape: &[usize],
    strides: &[usize],
) -> Result<(), FftError> {
    if shape.contains(&0) {
        return Ok(());
    }

    let mut axes = shape
        .iter()
        .zip(strides)
        .filter_map(|(extent, stride)| (*extent > 1).then_some((*stride, *extent)))
        .collect::<Vec<_>>();
    axes.sort_unstable_by_key(|(stride, _)| *stride);

    let mut span = 0usize;
    for (stride, extent) in axes {
        if stride <= span {
            return Err(FftError::OverlappingBindings);
        }
        span = span
            .checked_add(
                (extent - 1)
                    .checked_mul(stride)
                    .ok_or(FftError::SizeOverflow)?,
            )
            .ok_or(FftError::SizeOverflow)?;
    }
    Ok(())
}

fn validate_fft_shape(shape: &[usize], dim: usize) -> Result<(), FftError> {
    if dim >= shape.len() {
        return Err(FftError::AxisOutOfBounds {
            dim,
            rank: shape.len(),
        });
    }
    let n_fft = shape[dim];
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(FftError::InvalidFftLength { n_fft });
    }
    Ok(())
}

#[cube(launch)]
fn cfft_interleaved_shared_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    num_windows: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] mode: FftMode,
    #[comptime] normalization: FftNormalization,
) {
    let window_index = CUBE_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let input_re = input.view(InterleavedBatchSignalLayout::new(
        input,
        window_index,
        dim,
        0usize,
    ));
    let input_im = input.view(InterleavedBatchSignalLayout::new(
        input,
        window_index,
        dim,
        1usize,
    ));
    let mut shared_re = Shared::new_slice(n_fft);
    let mut shared_im = Shared::new_slice(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        shared_re[j] = input_re.read_checked(i);
        shared_im[j] = input_im.read_checked(i);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        mode,
    );

    let scale = match normalization {
        FftNormalization::None => F::new(1.0),
        FftNormalization::ByN => F::new(1.0) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0) / F::cast_from(n_fft).sqrt(),
    };
    {
        let mut output_re = output.view_mut(InterleavedBatchSignalLayout::new(
            &*output,
            window_index,
            dim,
            0usize,
        ));
        let mut k = UNIT_POS as usize;
        while k < n_fft {
            output_re.write_checked(k, shared_re[k] * scale);
            k += threads_per_cube;
        }
    }

    let mut output_im = output.view_mut(InterleavedBatchSignalLayout::new(
        &*output,
        window_index,
        dim,
        1usize,
    ));
    let mut k = UNIT_POS as usize;
    while k < n_fft {
        output_im.write_checked(k, shared_im[k] * scale);
        k += threads_per_cube;
    }
}

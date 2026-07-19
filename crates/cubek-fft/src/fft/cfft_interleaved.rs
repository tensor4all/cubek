use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, ComplexTensorHandle, FftError, FftNormalization,
    fft::{
        FftMode,
        cfft::{cfft_four_step_radix2_kernel, factor_four_step},
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

/// Launches a C32 FFT into an interleaved output tensor.
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
    if input.is_same_tensor(&output) {
        return Err(FftError::OverlappingBindings);
    }
    ensure_non_overlapping_output_layout(output.shape(), output.strides())?;

    normalization.scale_f32(plan.n_fft)?;

    output.ensure_unique_output()?;
    if plan.count == 0 {
        return Ok(());
    }

    if plan.n_fft <= max_shared_fft_n(client) {
        cfft_interleaved_shared_launch(client, input, output, dim, mode, normalization, plan)
    } else {
        cfft_interleaved_four_step_launch(client, input, output, dim, mode, normalization, plan)
    }
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
    let max_four_step_n = max_n.saturating_mul(max_n);
    if n_fft > max_four_step_n {
        return Err(FftError::InvalidFftLength { n_fft });
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

fn cfft_interleaved_shared_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: ComplexTensorBinding<'_, R>,
    output: ComplexTensorBinding<'_, R>,
    dim: usize,
    mode: FftMode,
    normalization: FftNormalization,
    plan: CfftPlan,
) -> Result<(), FftError> {
    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_cube = (plan.n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, plan.count, CubeDim::new_single());

    cfft_interleaved_shared_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        input.tensor().into_tensor_arg(),
        output.tensor().into_tensor_arg(),
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

fn cfft_interleaved_four_step_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: ComplexTensorBinding<'_, R>,
    output: ComplexTensorBinding<'_, R>,
    dim: usize,
    mode: FftMode,
    normalization: FftNormalization,
    plan: CfftPlan,
) -> Result<(), FftError> {
    let max_n = max_shared_fft_n(client);
    let max_four_step_n = max_n.saturating_mul(max_n);
    if plan.n_fft > max_four_step_n {
        return Err(FftError::InvalidFftLength { n_fft: plan.n_fft });
    }
    let (n1, n2) = factor_four_step(plan.n_fft, max_n);
    let total = plan
        .count
        .checked_mul(plan.n_fft)
        .ok_or(FftError::SizeOverflow)?;
    let total_u32 = u32::try_from(total).map_err(|_| FftError::SizeOverflow)?;
    let num_radix1_cubes = plan.count.checked_mul(n2).ok_or(FftError::SizeOverflow)?;
    let num_radix1_cubes_u32 =
        u32::try_from(num_radix1_cubes).map_err(|_| FftError::SizeOverflow)?;
    let num_radix2_cubes = plan.count.checked_mul(n1).ok_or(FftError::SizeOverflow)?;
    let num_radix2_cubes_u32 =
        u32::try_from(num_radix2_cubes).map_err(|_| FftError::SizeOverflow)?;
    let byte_len = total
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FftError::SizeOverflow)?;
    let shape = input.shape().to_vec();
    let dtype = input.dtype();
    let scratch_re =
        TensorHandle::<R>::new_contiguous(shape.clone(), client.empty(byte_len), dtype);
    let scratch_im = TensorHandle::<R>::new_contiguous(shape, client.empty(byte_len), dtype);
    let max_units = max_units_per_cube(client);

    {
        let threads_per_cube = (n1 / 2).clamp(1, max_units);
        let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
        let cube_count =
            cubecl::calculate_cube_count_elemwise(client, num_radix1_cubes, CubeDim::new_single());
        cfft_interleaved_four_step_radix1_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            input.tensor().into_tensor_arg(),
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            num_radix1_cubes_u32,
            n1,
            n2,
            n1.trailing_zeros() as usize,
            threads_per_cube,
            dim,
            mode,
        );
    }

    {
        let threads_per_cube = (n2 / 2).clamp(1, max_units);
        let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
        let cube_count =
            cubecl::calculate_cube_count_elemwise(client, num_radix2_cubes, CubeDim::new_single());
        cfft_four_step_radix2_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            num_radix2_cubes_u32,
            n1,
            n2,
            n2.trailing_zeros() as usize,
            threads_per_cube,
            dim,
            mode,
        );
    }

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, total, cube_dim);
    cfft_interleaved_four_step_transpose_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        scratch_re.binding().into_tensor_arg(),
        scratch_im.binding().into_tensor_arg(),
        output.tensor().into_tensor_arg(),
        total_u32,
        n1,
        n2,
        dim,
        normalization,
    );
    Ok(())
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
    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);
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
        FftNormalization::None => F::new(1.0_f32),
        FftNormalization::ByN => F::new(1.0_f32) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0_f32) / F::cast_from(n_fft).sqrt(),
    };
    {
        let output_re = output.view_mut(InterleavedBatchSignalLayout::new(
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

    let output_im = output.view_mut(InterleavedBatchSignalLayout::new(
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
    sync_cube();
}

/// First four-step pass over the strided N1 dimension of each C32 window.
/// Reads interleaved scalar pairs and writes split scratch with the fused
/// Cooley-Tukey twiddle.
#[cube(launch)]
fn cfft_interleaved_four_step_radix1_kernel<F: Float>(
    input: &Tensor<F>,
    scratch_re: &mut Tensor<F>,
    scratch_im: &mut Tensor<F>,
    num_cubes: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] log2_n1: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] mode: FftMode,
) {
    let cube_pos = CUBE_POS;
    if cube_pos >= num_cubes as usize {
        terminate!();
    }

    let window = cube_pos / n2;
    let n2_idx = cube_pos - window * n2;
    let input_re = input.view(InterleavedBatchSignalLayout::new(
        input, window, dim, 0usize,
    ));
    let input_im = input.view(InterleavedBatchSignalLayout::new(
        input, window, dim, 1usize,
    ));
    let scratch_re_view = scratch_re.view_mut(crate::layout::BatchSignalLayout::new(
        &*scratch_re,
        window,
        dim,
    ));
    let scratch_im_view = scratch_im.view_mut(crate::layout::BatchSignalLayout::new(
        &*scratch_im,
        window,
        dim,
    ));
    let mut shared_re = SharedMemory::<F>::new(n1);
    let mut shared_im = SharedMemory::<F>::new(n1);

    let mut i = UNIT_POS as usize;
    while i < n1 {
        let j = bit_reverse(i, log2_n1);
        let flat = i * n2 + n2_idx;
        shared_re[j] = input_re.read_checked(flat);
        shared_im[j] = input_im.read_checked(flat);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n1,
        log2_n1,
        threads_per_cube,
        mode,
    );

    let sign = F::new(mode.sign());
    let n_total = comptime![n1 * n2];
    let two_pi = F::new(2.0 * PI);
    let mut k1 = UNIT_POS as usize;
    while k1 < n1 {
        let theta = sign * two_pi * F::cast_from(k1 * n2_idx) / F::cast_from(n_total);
        let w_re = theta.cos();
        let w_im = theta.sin();
        let ar = shared_re[k1];
        let ai = shared_im[k1];
        let flat = k1 * n2 + n2_idx;
        scratch_re_view.write_checked(flat, w_re * ar - w_im * ai);
        scratch_im_view.write_checked(flat, w_re * ai + w_im * ar);
        k1 += threads_per_cube;
    }
}

/// Final four-step transpose writes adjacent interleaved output scalars and
/// applies the requested normalization as part of the global store.
#[cube(launch)]
fn cfft_interleaved_four_step_transpose_kernel<F: Float>(
    scratch_re: &Tensor<F>,
    scratch_im: &Tensor<F>,
    output: &mut Tensor<F>,
    total: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] dim: usize,
    #[comptime] normalization: FftNormalization,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }

    let n_fft = comptime![n1 * n2];
    let inner = pos % n_fft;
    let window = pos / n_fft;
    let scratch_re_view = scratch_re.view(crate::layout::BatchSignalLayout::new(
        scratch_re, window, dim,
    ));
    let scratch_im_view = scratch_im.view(crate::layout::BatchSignalLayout::new(
        scratch_im, window, dim,
    ));
    let k2 = inner / n1;
    let k1 = inner - k2 * n1;
    let src = k1 * n2 + k2;
    let scale = match normalization {
        FftNormalization::None => F::new(1.0_f32),
        FftNormalization::ByN => F::new(1.0_f32) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0_f32) / F::cast_from(n_fft).sqrt(),
    };
    {
        let output_re = output.view_mut(InterleavedBatchSignalLayout::new(
            &*output, window, dim, 0usize,
        ));
        output_re.write_checked(inner, scratch_re_view.read_checked(src) * scale);
    }
    let output_im = output.view_mut(InterleavedBatchSignalLayout::new(
        &*output, window, dim, 1usize,
    ));
    output_im.write_checked(inner, scratch_im_view.read_checked(src) * scale);
}

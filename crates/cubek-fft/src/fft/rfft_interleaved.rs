//! Real-valued FFT into interleaved C32 output.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, ComplexTensorHandle, FftError, FftNormalization,
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{ensure_packed_cfft_supported, max_shared_fft_n, max_units_per_cube},
        rfft_large::rfft_interleaved_large_launch,
    },
    interleaved_layout::InterleavedBatchSignalLayout,
    layout::BatchSignalLayout,
};

/// Runs a real F32 FFT over `signal` and returns an interleaved C32 spectrum.
pub fn rfft_interleaved<R: Runtime>(
    signal: TensorHandle<R>,
    dim: usize,
    normalization: FftNormalization,
) -> Result<ComplexTensorHandle<R>, FftError> {
    let shape = signal.shape().to_vec();
    let n_fft = validate_real_signal(signal.dtype, &shape, dim)?;
    normalization.scale_f32(n_fft)?;

    let client = R::client(&Default::default());
    ensure_packed_cfft_supported(&client, n_fft)?;
    let mut spectrum_shape = shape;
    spectrum_shape[dim] = n_fft / 2 + 1;
    let spectrum = ComplexTensorHandle::empty(&client, spectrum_shape, signal.dtype)?;

    rfft_interleaved_launch(&client, &signal, spectrum.binding(), dim, normalization)?;
    Ok(spectrum)
}

/// Launches a real F32 FFT into caller-provided interleaved C32 output.
pub fn rfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: &TensorHandle<R>,
    spectrum: ComplexTensorBinding<'_, R>,
    dim: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let signal_len = signal
        .shape()
        .get(dim)
        .copied()
        .ok_or(FftError::AxisOutOfBounds {
            dim,
            rank: signal.shape().len(),
        })?;
    rfft_interleaved_launch_padded(client, signal, spectrum, dim, signal_len, normalization)
}

/// Launches an interleaved RFFT while treating samples at `signal_len..n_fft` as zero.
pub fn rfft_interleaved_launch_padded<R: Runtime>(
    client: &ComputeClient<R>,
    signal: &TensorHandle<R>,
    spectrum: ComplexTensorBinding<'_, R>,
    dim: usize,
    signal_len: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let plan = rfft_plan(signal, &spectrum, dim, signal_len)?;
    normalization.scale_f32(plan.n_fft)?;
    ensure_packed_cfft_supported(client, plan.n_fft)?;

    spectrum.ensure_unique_output()?;
    if plan.count == 0 {
        return Ok(());
    }
    if plan.n_fft > max_shared_fft_n(client) {
        return rfft_interleaved_large_launch(
            client,
            signal,
            spectrum,
            dim,
            signal_len,
            normalization,
            plan.n_fft,
            plan.count,
        );
    }

    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_cube = (plan.n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, plan.count, CubeDim::new_single());

    rfft_interleaved_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        signal.clone().binding().into_tensor_arg(),
        spectrum.tensor().into_tensor_arg(),
        plan.count_u32,
        signal_len as u32,
        plan.n_fft,
        log2_n,
        threads_per_cube,
        dim,
        normalization,
    );
    Ok(())
}

struct RfftPlan {
    n_fft: usize,
    count: usize,
    count_u32: u32,
}

fn rfft_plan<R: Runtime>(
    signal: &TensorHandle<R>,
    spectrum: &ComplexTensorBinding<'_, R>,
    dim: usize,
    signal_len: usize,
) -> Result<RfftPlan, FftError> {
    let signal_shape = signal.shape();
    validate_signal_dtype_and_axis(signal.dtype, signal_shape, dim)?;
    if spectrum.dtype() != f32::as_type_native_unchecked().storage_type() {
        return Err(FftError::UnsupportedDtype {
            actual: spectrum.dtype(),
        });
    }
    if spectrum.shape().len() != signal_shape.len() {
        return Err(FftError::ShapeMismatch {
            name: "spectrum",
            actual: spectrum.shape().to_vec(),
            expected: signal_shape.to_vec(),
        });
    }

    let n_freq = spectrum.shape()[dim];
    if n_freq < 2 {
        return Err(FftError::InvalidFftLength { n_fft: 0 });
    }
    let n_fft = n_freq
        .checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .ok_or(FftError::SizeOverflow)?;
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(FftError::InvalidFftLength { n_fft });
    }

    let mut expected_shape = signal_shape.to_vec();
    expected_shape[dim] = n_freq;
    if spectrum.shape() != expected_shape {
        return Err(FftError::ShapeMismatch {
            name: "spectrum",
            actual: spectrum.shape().to_vec(),
            expected: expected_shape,
        });
    }
    ensure_non_overlapping_output_layout(spectrum.shape(), spectrum.strides())?;
    if signal_len > signal_shape[dim] {
        return Err(FftError::InvalidLength {
            name: "signal_len",
            value: signal_len,
            min: 0,
            max: signal_shape[dim],
        });
    }
    if signal_len > n_fft {
        return Err(FftError::InvalidLength {
            name: "signal_len",
            value: signal_len,
            min: 0,
            max: n_fft,
        });
    }

    let count = signal_shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != dim)
        .try_fold(1usize, |count, (_, extent)| {
            count.checked_mul(*extent).ok_or(FftError::SizeOverflow)
        })?;
    let count_u32 = u32::try_from(count).map_err(|_| FftError::SizeOverflow)?;
    Ok(RfftPlan {
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

fn validate_real_signal(
    dtype: StorageType,
    shape: &[usize],
    dim: usize,
) -> Result<usize, FftError> {
    validate_signal_dtype_and_axis(dtype, shape, dim)?;
    let n_fft = shape[dim];
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(FftError::InvalidFftLength { n_fft });
    }
    Ok(n_fft)
}

fn validate_signal_dtype_and_axis(
    dtype: StorageType,
    shape: &[usize],
    dim: usize,
) -> Result<(), FftError> {
    if dtype != f32::as_type_native_unchecked().storage_type() {
        return Err(FftError::UnsupportedDtype { actual: dtype });
    }
    shape.get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: shape.len(),
    })?;
    Ok(())
}

#[cube(launch)]
fn rfft_interleaved_kernel<F: Float>(
    signal: &Tensor<F>,
    spectrum: &mut Tensor<F>,
    num_windows: u32,
    signal_len: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] normalization: FftNormalization,
) {
    let window_index = CUBE_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let signal_view = signal.view(BatchSignalLayout::new(signal, window_index, dim));
    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        let active = i < signal_len as usize;
        let src = select(active, i, 0);
        shared_re[j] = select(active, signal_view.read_checked(src), F::new(0.0_f32));
        shared_im[j] = F::new(0.0_f32);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Forward,
    );

    let scale = match normalization {
        FftNormalization::None => F::new(1.0_f32),
        FftNormalization::ByN => F::new(1.0_f32) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0_f32) / F::cast_from(n_fft).sqrt(),
    };
    let n_freq = comptime![n_fft / 2 + 1];
    {
        let spectrum_re = spectrum.view_mut(InterleavedBatchSignalLayout::new(
            &*spectrum,
            window_index,
            dim,
            0usize,
        ));
        let mut k = UNIT_POS as usize;
        while k < n_freq {
            spectrum_re.write_checked(k, shared_re[k] * scale);
            k += threads_per_cube;
        }
    }
    let spectrum_im = spectrum.view_mut(InterleavedBatchSignalLayout::new(
        &*spectrum,
        window_index,
        dim,
        1usize,
    ));
    let mut k = UNIT_POS as usize;
    while k < n_freq {
        spectrum_im.write_checked(k, shared_im[k] * scale);
        k += threads_per_cube;
    }
}

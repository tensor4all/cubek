//! Inverse real-valued FFT from interleaved C32 input.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, ComplexTensorHandle, FftError, FftNormalization,
    complex::ensure_unique_output,
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
    },
    interleaved_layout::InterleavedBatchSignalLayout,
    layout::BatchSignalLayout,
};

/// Runs an inverse real F32 FFT from an interleaved C32 half-spectrum.
pub fn irfft_interleaved<R: Runtime>(
    spectrum: ComplexTensorHandle<R>,
    dim: usize,
    normalization: FftNormalization,
) -> Result<TensorHandle<R>, FftError> {
    let client = R::client(&Default::default());
    let spectrum_shape = spectrum.shape();
    let n_freq = *spectrum_shape.get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: spectrum_shape.len(),
    })?;
    let n_fft = n_freq
        .checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .ok_or(FftError::SizeOverflow)?;
    let mut signal_shape = spectrum_shape.to_vec();
    signal_shape[dim] = n_fft;
    let plan = irfft_plan(
        &spectrum.binding(),
        &signal_shape,
        spectrum.dtype(),
        dim,
        n_freq,
    )?;
    if plan.n_fft > max_shared_fft_n(&client) {
        return Err(FftError::InvalidFftLength { n_fft: plan.n_fft });
    }

    let elements = signal_shape.iter().try_fold(1usize, |total, extent| {
        total.checked_mul(*extent).ok_or(FftError::SizeOverflow)
    })?;
    let byte_len = elements
        .checked_mul(spectrum.dtype().size())
        .ok_or(FftError::SizeOverflow)?;
    let signal =
        TensorHandle::new_contiguous(signal_shape, client.empty(byte_len), spectrum.dtype());
    irfft_interleaved_launch_padded(
        &client,
        spectrum.binding(),
        &signal,
        dim,
        n_freq,
        normalization,
    )?;
    Ok(signal)
}

/// Launches an inverse real F32 FFT into caller-provided real output.
pub fn irfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum: ComplexTensorBinding<'_, R>,
    signal: &TensorHandle<R>,
    dim: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let spec_bins = *spectrum.shape().get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: spectrum.shape().len(),
    })?;
    irfft_interleaved_launch_padded(client, spectrum, signal, dim, spec_bins, normalization)
}

/// Launches an interleaved IRFFT while treating bins at `spec_bins..n_freq` as zero.
pub fn irfft_interleaved_launch_padded<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum: ComplexTensorBinding<'_, R>,
    signal: &TensorHandle<R>,
    dim: usize,
    spec_bins: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let plan = irfft_plan(&spectrum, signal.shape(), signal.dtype, dim, spec_bins)?;
    ensure_non_overlapping_output_layout(signal.shape(), signal.strides())?;
    ensure_unique_output(signal)?;
    if plan.count == 0 {
        return Ok(());
    }
    if plan.n_fft > max_shared_fft_n(client) {
        return Err(FftError::InvalidFftLength { n_fft: plan.n_fft });
    }

    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_cube = (plan.n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, plan.count, CubeDim::new_single());
    irfft_interleaved_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        spectrum.tensor().into_tensor_arg(),
        signal.clone().binding().into_tensor_arg(),
        plan.count_u32,
        spec_bins as u32,
        plan.n_fft,
        log2_n,
        threads_per_cube,
        dim,
        normalization,
    );
    Ok(())
}

struct IrfftPlan {
    n_fft: usize,
    count: usize,
    count_u32: u32,
}

fn irfft_plan<R: Runtime>(
    spectrum: &ComplexTensorBinding<'_, R>,
    signal_shape: &[usize],
    signal_dtype: StorageType,
    dim: usize,
    spec_bins: usize,
) -> Result<IrfftPlan, FftError> {
    if spectrum.dtype() != f32::as_type_native_unchecked().storage_type() {
        return Err(FftError::UnsupportedDtype {
            actual: spectrum.dtype(),
        });
    }
    if signal_dtype != f32::as_type_native_unchecked().storage_type() {
        return Err(FftError::UnsupportedDtype {
            actual: signal_dtype,
        });
    }
    if signal_shape.len() != spectrum.shape().len() {
        return Err(FftError::ShapeMismatch {
            name: "signal",
            actual: signal_shape.to_vec(),
            expected: spectrum.shape().to_vec(),
        });
    }
    let n_fft = *signal_shape.get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: signal_shape.len(),
    })?;
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(FftError::InvalidFftLength { n_fft });
    }
    let n_freq = n_fft / 2 + 1;
    let mut expected_spectrum_shape = signal_shape.to_vec();
    expected_spectrum_shape[dim] = spectrum.shape()[dim];
    if spectrum.shape() != expected_spectrum_shape {
        return Err(FftError::ShapeMismatch {
            name: "spectrum",
            actual: spectrum.shape().to_vec(),
            expected: expected_spectrum_shape,
        });
    }
    if spec_bins == 0 || spec_bins > spectrum.shape()[dim] || spec_bins > n_freq {
        return Err(FftError::InvalidLength {
            name: "spec_bins",
            value: spec_bins,
            min: 1,
            max: spectrum.shape()[dim].min(n_freq),
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
    Ok(IrfftPlan {
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

#[cube(launch)]
fn irfft_interleaved_kernel<F: Float>(
    spectrum: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    spec_bins: u32,
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

    let spectrum_re = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum,
        window_index,
        dim,
        0usize,
    ));
    let spectrum_im = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum,
        window_index,
        dim,
        1usize,
    ));
    let mut signal_view = signal.view_mut(BatchSignalLayout::new(&*signal, window_index, dim));
    let mut shared_re = Shared::new_slice(n_fft);
    let mut shared_im = Shared::new_slice(n_fft);
    let n_freq = comptime![n_fft / 2 + 1];

    let mut k = UNIT_POS as usize;
    while k < n_fft {
        let dst = bit_reverse(k, log2_n);
        let src_bin = select(k < n_freq, k, n_fft - k);
        let active = src_bin < spec_bins as usize;
        let src_bin = select(active, src_bin, 0);
        let im_sign = select(k < n_freq, F::new(1.0), F::new(-1.0));
        shared_re[dst] = select(active, spectrum_re.read_checked(src_bin), F::new(0.0));
        shared_im[dst] = select(
            active,
            spectrum_im.read_checked(src_bin) * im_sign,
            F::new(0.0),
        );
        k += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Inverse,
    );

    let scale = match normalization {
        FftNormalization::None => F::new(1.0),
        FftNormalization::ByN => F::new(1.0) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0) / F::cast_from(n_fft).sqrt(),
    };
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        signal_view.write_checked(i, shared_re[i] * scale);
        i += threads_per_cube;
    }
}

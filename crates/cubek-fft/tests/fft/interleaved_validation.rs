use cubecl::{Runtime, TestRuntime, frontend::CubePrimitive, std::tensor::TensorHandle};
use cubek_fft::{
    ComplexTensorHandle, FftError, FftNormalization, irfft_interleaved, irfft_interleaved_launch,
    rfft_interleaved, rfft_interleaved_launch,
};

#[test]
fn contiguous_c32_uses_two_adjacent_scalars_per_logical_element() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let complex = ComplexTensorHandle::<TestRuntime>::empty(&client, vec![2, 3], dtype).unwrap();
    assert_eq!(complex.shape(), &[2, 3]);
    assert_eq!(complex.strides(), &[3, 1]);
    assert_eq!(complex.scalar_strides(), &[6, 2]);
    assert_eq!(complex.physical_scalar_len(), 12);
}

#[test]
fn c32_rejects_wrong_dtype_and_short_buffer() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_dtype = f32::as_type_native_unchecked().storage_type();
    let f64_dtype = f64::as_type_native_unchecked().storage_type();
    let wrong = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        vec![4],
        client.empty(8 * f64_dtype.size()),
        f64_dtype,
    );
    assert!(matches!(wrong, Err(FftError::UnsupportedDtype { .. })));
    let short = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        vec![4],
        client.empty(7 * f32_dtype.size()),
        f32_dtype,
    );
    assert!(matches!(short, Err(FftError::InsufficientBuffer { .. })));
}

#[test]
fn normalization_scales_are_direction_independent() {
    assert_eq!(FftNormalization::None.scale_f32(16).unwrap(), 1.0);
    assert_eq!(FftNormalization::ByN.scale_f32(16).unwrap(), 1.0 / 16.0);
    assert_eq!(FftNormalization::Ortho.scale_f32(16).unwrap(), 0.25);
}

#[test]
fn c32_metadata_errors_are_typed() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let rank = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![2],
        vec![],
        client.empty(8 * dtype.size()),
        dtype,
    );
    assert!(matches!(rank, Err(FftError::RankMismatch { .. })));

    let misaligned = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        vec![2],
        client.empty(5 * dtype.size()).offset_start(1),
        dtype,
    );
    assert!(matches!(misaligned, Err(FftError::MisalignedBuffer { .. })));

    let stride_overflow = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![2],
        vec![usize::MAX],
        client.empty(4 * dtype.size()),
        dtype,
    );
    assert!(matches!(
        stride_overflow,
        Err(FftError::StrideOverflow { axis: 0 })
    ));

    let extent_overflow = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![usize::MAX, 2],
        vec![1, 1],
        client.empty(4 * dtype.size()),
        dtype,
    );
    assert!(matches!(extent_overflow, Err(FftError::SizeOverflow)));
}

#[test]
fn zero_sized_c32_shape_has_no_physical_scalars() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let complex =
        ComplexTensorHandle::<TestRuntime>::new_contiguous(vec![0, 3], client.empty(0), dtype)
            .unwrap();
    assert_eq!(complex.physical_scalar_len(), 0);
}

#[test]
fn non_contiguous_c32_extent_includes_the_last_imaginary_scalar() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let complex = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![2, 3],
        vec![5, 1],
        client.empty(16 * dtype.size()),
        dtype,
    )
    .unwrap();
    assert_eq!(complex.scalar_strides(), &[10, 2]);
    assert_eq!(complex.physical_scalar_len(), 16);
}

fn first_unsupported_real_fft_n() -> Option<usize> {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max_elems =
        client.properties().hardware.max_shared_memory_size / (2 * core::mem::size_of::<f32>());
    let max_shared = if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() >> 1
    };
    max_shared.checked_mul(max_shared)?.checked_mul(4)
}

#[test]
fn oversized_rfft_is_rejected_for_allocating_and_caller_owned_apis_without_allocating_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let Some(n_fft) = first_unsupported_real_fft_n() else {
        // The CPU backend uses system RAM as its shared-memory limit, so its
        // reported maximum can exceed every FFT length representable by usize.
        return;
    };
    let signal = TensorHandle::new_contiguous(vec![0, n_fft], client.empty(0), dtype);

    assert!(matches!(
        rfft_interleaved(signal.clone(), 1, FftNormalization::None),
        Err(FftError::FftLengthExceedsDeviceLimit { .. })
    ));

    let spectrum =
        ComplexTensorHandle::new_contiguous(vec![0, n_fft / 2 + 1], client.empty(0), dtype)
            .unwrap();
    assert!(matches!(
        rfft_interleaved_launch(
            &client,
            &signal,
            spectrum.binding(),
            1,
            FftNormalization::None,
        ),
        Err(FftError::FftLengthExceedsDeviceLimit { .. })
    ));
}

#[test]
fn oversized_irfft_is_rejected_for_allocating_and_caller_owned_apis_without_allocating_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let Some(n_fft) = first_unsupported_real_fft_n() else {
        // See the RFFT case above: there is no representable unsupported
        // length to exercise for a backend with a saturating device limit.
        return;
    };
    let spectrum =
        ComplexTensorHandle::new_contiguous(vec![0, n_fft / 2 + 1], client.empty(0), dtype)
            .unwrap();

    assert!(matches!(
        irfft_interleaved(spectrum.clone(), 1, FftNormalization::ByN),
        Err(FftError::FftLengthExceedsDeviceLimit { .. })
    ));

    let signal = TensorHandle::new_contiguous(vec![0, n_fft], client.empty(0), dtype);
    assert!(matches!(
        irfft_interleaved_launch(
            &client,
            spectrum.binding(),
            &signal,
            1,
            FftNormalization::ByN,
        ),
        Err(FftError::FftLengthExceedsDeviceLimit { .. })
    ));
}

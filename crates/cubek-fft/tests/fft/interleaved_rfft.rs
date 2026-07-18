use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive};
use cubek_fft::eval::cpu_reference::rfft_ref;
use cubek_fft::{
    ComplexTensorHandle, FftError, FftNormalization, irfft_interleaved, rfft_interleaved,
    rfft_interleaved_launch, rfft_interleaved_launch_padded,
};
use cubek_test_utils::{HostData, HostDataType, HostDataVec};

fn real_tensor(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    values: &[f32],
) -> TensorHandle<TestRuntime> {
    TensorHandle::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(values)),
        f32::as_type_native_unchecked().storage_type(),
    )
}

fn scalar_buffer(
    client: &ComputeClient<TestRuntime>,
    tensor: ComplexTensorHandle<TestRuntime>,
) -> Vec<f32> {
    let raw = tensor.into_raw_parts();
    f32::from_bytes(&client.read_one(raw.handle).unwrap()).to_vec()
}

fn to_f32(data: HostData) -> Vec<f32> {
    match data.data {
        HostDataVec::F32(values) => values,
        _ => panic!("expected F32 host data"),
    }
}

fn expected_interleaved(signal: HostData, dim: usize, normalization: FftNormalization) -> Vec<f32> {
    let n_fft = signal.shape.as_slice()[dim];
    let scale = normalization.scale_f32(n_fft).unwrap();
    let (re, im) = rfft_ref(&signal, dim, None);
    to_f32(re)
        .into_iter()
        .zip(to_f32(im))
        .flat_map(|(re, im)| [re * scale, im * scale])
        .collect()
}

fn values_for(shape: &[usize]) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .map(|index| ((index as f32 + 0.25) * 0.37).sin())
        .collect()
}

fn assert_scalars_approx(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "scalar {index}: got {actual}, expected {expected}"
        );
    }
}

fn run_allocating_case(shape: Vec<usize>, dim: usize, normalization: FftNormalization) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let values = values_for(&shape);
    let signal = real_tensor(&client, shape.clone(), &values);
    let expected = expected_interleaved(
        HostData::from_tensor_handle(&client, signal.clone(), HostDataType::F32),
        dim,
        normalization,
    );

    let spectrum = rfft_interleaved(signal, dim, normalization).unwrap();
    let mut expected_shape = shape;
    expected_shape[dim] = expected_shape[dim] / 2 + 1;
    assert_eq!(spectrum.shape(), expected_shape);
    assert_scalars_approx(&scalar_buffer(&client, spectrum), &expected);
}

#[cfg(feature = "heavy")]
fn first_large_n(client: &ComputeClient<TestRuntime>) -> usize {
    let max_elems =
        client.properties().hardware.max_shared_memory_size / (2 * core::mem::size_of::<f32>());
    let max_shared_fft_n = if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() >> 1
    };
    2 * max_shared_fft_n
}

#[cfg(feature = "heavy")]
fn run_large_round_trip(shape: Vec<usize>, dim: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let values = values_for(&shape);
    let signal = real_tensor(&client, shape, &values);
    let spectrum = rfft_interleaved(signal, dim, FftNormalization::None).unwrap();
    let reconstructed = irfft_interleaved(spectrum, dim, FftNormalization::ByN).unwrap();
    let reconstructed_bytes = client.read_one(reconstructed.handle).unwrap();
    let reconstructed = f32::from_bytes(&reconstructed_bytes);
    for (index, (actual, expected)) in reconstructed.iter().zip(&values).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.04,
            "sample {index}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn rfft_interleaved_axis_last_matches_reference() {
    run_allocating_case(vec![2, 8], 1, FftNormalization::None);
}

#[test]
fn rfft_interleaved_axis_zero_and_middle_match_reference_with_trailing_batches() {
    run_allocating_case(vec![8, 2, 3], 0, FftNormalization::None);
    run_allocating_case(vec![2, 8, 3], 1, FftNormalization::None);
}

#[test]
fn rfft_interleaved_applies_all_normalizations_at_direct_c32_stores() {
    for normalization in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        run_allocating_case(vec![2, 8, 3], 1, normalization);
    }
}

#[test]
fn rfft_interleaved_padded_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let virtual_shape = vec![2, 5, 3];
    let mut padded_shape = virtual_shape.clone();
    padded_shape[1] = 8;
    let mut spectrum_shape = virtual_shape.clone();
    spectrum_shape[1] = 5;

    let virtual_signal = real_tensor(&client, virtual_shape.clone(), &values_for(&virtual_shape));
    let mut padded_values = vec![0.0; padded_shape.iter().product()];
    for batch_before in 0..virtual_shape[0] {
        for sample in 0..virtual_shape[1] {
            for batch_after in 0..virtual_shape[2] {
                padded_values
                    [(batch_before * padded_shape[1] + sample) * padded_shape[2] + batch_after] =
                    values_for(&virtual_shape)[(batch_before * virtual_shape[1] + sample)
                        * virtual_shape[2]
                        + batch_after];
            }
        }
    }
    let padded_signal = real_tensor(&client, padded_shape, &padded_values);
    let virtual_spectrum =
        ComplexTensorHandle::empty(&client, spectrum_shape.clone(), dtype).unwrap();
    let materialized_spectrum = ComplexTensorHandle::empty(&client, spectrum_shape, dtype).unwrap();

    rfft_interleaved_launch_padded(
        &client,
        &virtual_signal,
        virtual_spectrum.binding(),
        1,
        virtual_shape[1],
        FftNormalization::Ortho,
    )
    .unwrap();
    rfft_interleaved_launch(
        &client,
        &padded_signal,
        materialized_spectrum.binding(),
        1,
        FftNormalization::Ortho,
    )
    .unwrap();

    assert_scalars_approx(
        &scalar_buffer(&client, virtual_spectrum),
        &scalar_buffer(&client, materialized_spectrum),
    );
}

#[test]
fn rfft_interleaved_launch_rejects_overlapping_output_layout() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let signal = real_tensor(&client, vec![2, 8], &values_for(&[2, 8]));
    let spectrum = ComplexTensorHandle::new_strided(
        vec![2, 5],
        vec![0, 1],
        client.empty(10 * dtype.size()),
        dtype,
    )
    .unwrap();

    assert!(matches!(
        rfft_interleaved_launch(
            &client,
            &signal,
            spectrum.binding(),
            1,
            FftNormalization::None,
        ),
        Err(FftError::OverlappingBindings)
    ));
}

#[test]
#[cfg(feature = "heavy")]
fn interleaved_rfft_and_irfft_first_large_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_large_n(&client);
    run_large_round_trip(vec![2, n_fft], 1);
}

#[test]
#[cfg(feature = "heavy")]
fn interleaved_rfft_and_irfft_batched_large_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_large_n(&client);
    run_large_round_trip(vec![3, n_fft], 1);
}

#[test]
#[cfg(feature = "heavy")]
fn interleaved_rfft_and_irfft_strided_axis_large_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_large_n(&client);
    run_large_round_trip(vec![2, n_fft, 3], 1);
}

#[test]
#[cfg(feature = "heavy")]
fn interleaved_rfft_large_virtual_padding_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let n_fft = first_large_n(&client);
    let virtual_shape = vec![2, n_fft / 2, 3];
    let mut padded_shape = virtual_shape.clone();
    padded_shape[1] = n_fft;
    let mut spectrum_shape = virtual_shape.clone();
    spectrum_shape[1] = n_fft / 2 + 1;
    let virtual_values = values_for(&virtual_shape);
    let virtual_signal = real_tensor(&client, virtual_shape.clone(), &virtual_values);
    let mut padded_values = vec![0.0; padded_shape.iter().product()];
    for before in 0..virtual_shape[0] {
        for sample in 0..virtual_shape[1] {
            for after in 0..virtual_shape[2] {
                padded_values[(before * n_fft + sample) * virtual_shape[2] + after] =
                    virtual_values[(before * virtual_shape[1] + sample) * virtual_shape[2] + after];
            }
        }
    }
    let padded_signal = real_tensor(&client, padded_shape, &padded_values);
    let virtual_spectrum =
        ComplexTensorHandle::empty(&client, spectrum_shape.clone(), dtype).unwrap();
    let materialized_spectrum = ComplexTensorHandle::empty(&client, spectrum_shape, dtype).unwrap();

    rfft_interleaved_launch_padded(
        &client,
        &virtual_signal,
        virtual_spectrum.binding(),
        1,
        virtual_shape[1],
        FftNormalization::Ortho,
    )
    .unwrap();
    rfft_interleaved_launch(
        &client,
        &padded_signal,
        materialized_spectrum.binding(),
        1,
        FftNormalization::Ortho,
    )
    .unwrap();

    assert_scalars_approx(
        &scalar_buffer(&client, virtual_spectrum),
        &scalar_buffer(&client, materialized_spectrum),
    );
}

use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive};
use cubek_fft::eval::cpu_reference::irfft_ref;
use cubek_fft::{
    ComplexTensorHandle, FftError, FftNormalization, irfft_interleaved, irfft_interleaved_launch,
    irfft_interleaved_launch_padded,
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

fn spectrum_values(shape: &[usize]) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .flat_map(|index| {
            let value = ((index as f32 + 0.25) * 0.37).sin();
            [value, value * 0.5 - 0.25]
        })
        .collect()
}

fn expected_signal(
    client: &ComputeClient<TestRuntime>,
    spectrum: ComplexTensorHandle<TestRuntime>,
    dim: usize,
    normalization: FftNormalization,
) -> Vec<f32> {
    let shape = spectrum.shape().to_vec();
    let n_fft = (shape[dim] - 1) * 2;
    let scalars = scalar_buffer(client, spectrum);
    let re = scalars.iter().step_by(2).copied().collect::<Vec<_>>();
    let im = scalars
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .collect::<Vec<_>>();
    let re = real_tensor(client, shape.clone(), &re);
    let im = real_tensor(client, shape, &im);
    let by_n = to_f32(irfft_ref(
        &HostData::from_tensor_handle(client, re, HostDataType::F32),
        &HostData::from_tensor_handle(client, im, HostDataType::F32),
        dim,
        None,
    ));
    let multiplier = match normalization {
        FftNormalization::ByN => 1.0,
        FftNormalization::None => n_fft as f32,
        FftNormalization::Ortho => (n_fft as f32).sqrt(),
    };
    by_n.into_iter().map(|value| value * multiplier).collect()
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
    let dtype = f32::as_type_native_unchecked().storage_type();
    let expected = expected_signal(
        &client,
        ComplexTensorHandle::<TestRuntime>::new_contiguous(
            shape.clone(),
            client.create_from_slice(f32::as_bytes(&spectrum_values(&shape))),
            dtype,
        )
        .unwrap(),
        dim,
        normalization,
    );
    let spectrum = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        shape.clone(),
        client.create_from_slice(f32::as_bytes(&spectrum_values(&shape))),
        dtype,
    )
    .unwrap();

    let signal = irfft_interleaved(spectrum, dim, normalization).unwrap();
    let mut expected_shape = shape;
    expected_shape[dim] = (expected_shape[dim] - 1) * 2;
    assert_eq!(signal.shape().as_slice(), expected_shape);
    assert_scalars_approx(
        &f32::from_bytes(&client.read_one(signal.handle).unwrap()),
        &expected,
    );
}

#[test]
fn irfft_interleaved_axis_last_matches_reference() {
    run_allocating_case(vec![2, 5], 1, FftNormalization::ByN);
}

#[test]
fn irfft_interleaved_axis_zero_and_middle_match_reference_with_trailing_batches() {
    run_allocating_case(vec![5, 2, 3], 0, FftNormalization::ByN);
    run_allocating_case(vec![2, 5, 3], 1, FftNormalization::ByN);
}

#[test]
fn irfft_interleaved_applies_all_normalizations_at_final_real_stores() {
    for normalization in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        run_allocating_case(vec![2, 5, 3], 1, normalization);
    }
}

#[test]
fn irfft_interleaved_padded_dc_only_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let virtual_shape = vec![2, 1, 3];
    let mut materialized_shape = virtual_shape.clone();
    materialized_shape[1] = 5;
    let signal_shape = vec![2, 8, 3];
    let virtual_spectrum = ComplexTensorHandle::new_contiguous(
        virtual_shape.clone(),
        client.create_from_slice(f32::as_bytes(&spectrum_values(&virtual_shape))),
        dtype,
    )
    .unwrap();
    let mut materialized_values = vec![0.0; materialized_shape.iter().product::<usize>() * 2];
    for before in 0..virtual_shape[0] {
        for after in 0..virtual_shape[2] {
            let virtual_offset = (before * virtual_shape[1] * virtual_shape[2] + after) * 2;
            let materialized_offset =
                (before * materialized_shape[1] * materialized_shape[2] + after) * 2;
            materialized_values[materialized_offset..materialized_offset + 2].copy_from_slice(
                &spectrum_values(&virtual_shape)[virtual_offset..virtual_offset + 2],
            );
        }
    }
    let materialized_spectrum = ComplexTensorHandle::new_contiguous(
        materialized_shape,
        client.create_from_slice(f32::as_bytes(&materialized_values)),
        dtype,
    )
    .unwrap();
    let virtual_signal = real_tensor(&client, signal_shape.clone(), &vec![0.0; 48]);
    let materialized_signal = real_tensor(&client, signal_shape, &vec![0.0; 48]);

    irfft_interleaved_launch_padded(
        &client,
        virtual_spectrum.binding(),
        &virtual_signal,
        1,
        1,
        FftNormalization::Ortho,
    )
    .unwrap();
    irfft_interleaved_launch(
        &client,
        materialized_spectrum.binding(),
        &materialized_signal,
        1,
        FftNormalization::Ortho,
    )
    .unwrap();

    assert_scalars_approx(
        &f32::from_bytes(&client.read_one(virtual_signal.handle).unwrap()),
        &f32::from_bytes(&client.read_one(materialized_signal.handle).unwrap()),
    );
}

#[test]
fn irfft_interleaved_launch_rejects_non_f32_output() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = vec![2, 5];
    let spectrum = ComplexTensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.create_from_slice(f32::as_bytes(&spectrum_values(&spectrum_shape))),
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();
    let output = TensorHandle::new_contiguous(
        vec![2, 8],
        client.empty(16 * i32::as_type_native_unchecked().storage_type().size()),
        i32::as_type_native_unchecked().storage_type(),
    );

    assert!(matches!(
        irfft_interleaved_launch(
            &client,
            spectrum.binding(),
            &output,
            1,
            FftNormalization::ByN,
        ),
        Err(FftError::UnsupportedDtype { .. })
    ));
}

#[test]
fn irfft_interleaved_launch_rejects_shared_output_allocation() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let allocation = client.empty(20 * dtype.size());
    let spectrum =
        ComplexTensorHandle::new_contiguous(vec![2, 5], allocation.clone(), dtype).unwrap();
    let output = TensorHandle::new_contiguous(vec![2, 8], allocation, dtype);

    assert!(matches!(
        irfft_interleaved_launch(
            &client,
            spectrum.binding(),
            &output,
            1,
            FftNormalization::ByN,
        ),
        Err(FftError::OverlappingBindings)
    ));
}

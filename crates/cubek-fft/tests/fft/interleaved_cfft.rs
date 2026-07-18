use cubecl::{CubeElement, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive};
use cubek_fft::{
    ComplexTensorHandle, FftError, FftMode, FftNormalization, cfft_interleaved,
    cfft_interleaved_launch,
};

fn contiguous_c32(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    values: &[f32],
) -> ComplexTensorHandle<TestRuntime> {
    let dtype = f32::as_type_native_unchecked().storage_type();
    ComplexTensorHandle::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(values)),
        dtype,
    )
    .unwrap()
}

fn scalar_buffer(
    client: &ComputeClient<TestRuntime>,
    tensor: ComplexTensorHandle<TestRuntime>,
) -> Vec<f32> {
    let raw = tensor.into_raw_parts();
    f32::from_bytes(&client.read_one(raw.handle).unwrap()).to_vec()
}

fn assert_complex_scalars_approx(
    client: &ComputeClient<TestRuntime>,
    tensor: ComplexTensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) {
    let actual = scalar_buffer(client, tensor);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= epsilon,
            "scalar {index}: got {actual}, expected {expected}"
        );
    }
}

fn assert_logical_c32_approx(
    client: &ComputeClient<TestRuntime>,
    tensor: ComplexTensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) {
    let shape = tensor.shape().to_vec();
    let scalar_strides = tensor.scalar_strides().to_vec();
    let scalars = scalar_buffer(client, tensor);
    for logical in 0..shape.iter().product::<usize>() {
        let mut remaining = logical;
        let mut scalar_index = 0;
        for axis in (0..shape.len()).rev() {
            let coord = remaining % shape[axis];
            remaining /= shape[axis];
            scalar_index += coord * scalar_strides[axis];
        }
        for component in 0..2 {
            let actual = scalars[scalar_index + component];
            let expected = expected[logical * 2 + component];
            assert!(
                (actual - expected).abs() <= epsilon,
                "logical {logical}, component {component}: got {actual}, expected {expected}"
            );
        }
    }
}

fn values_for(shape: &[usize]) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .flat_map(|i| [i as f32 + 0.25, -(i as f32) + 0.5])
        .collect()
}

fn run_round_trip(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dim: usize,
    normalization: FftNormalization,
    epsilon: f32,
) {
    let values = values_for(&shape);
    let input = contiguous_c32(client, shape, &values);
    let forward_normalization = match normalization {
        FftNormalization::ByN => FftNormalization::None,
        normalization => normalization,
    };
    let spectrum = cfft_interleaved(input, dim, FftMode::Forward, forward_normalization).unwrap();
    let inverse_normalization = match normalization {
        FftNormalization::None => FftNormalization::ByN,
        normalization => normalization,
    };
    let result = cfft_interleaved(spectrum, dim, FftMode::Inverse, inverse_normalization).unwrap();
    assert_complex_scalars_approx(client, result, &values, epsilon);
}

fn round_trip(shape: Vec<usize>, dim: usize, normalization: FftNormalization) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_round_trip(&client, shape, dim, normalization, 1e-4);
}

fn test_max_shared_fft_n(client: &ComputeClient<TestRuntime>) -> usize {
    let max_elems =
        client.properties().hardware.max_shared_memory_size / (2 * core::mem::size_of::<f32>());
    if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() >> 1
    }
}

#[cfg(feature = "heavy")]
fn first_four_step_n(client: &ComputeClient<TestRuntime>) -> usize {
    2 * test_max_shared_fft_n(client)
}

#[test]
fn cfft_interleaved_small_round_trip_preserves_c32_order() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let values: Vec<f32> = (0..8)
        .flat_map(|i| [i as f32 + 0.25, -(i as f32)])
        .collect();
    let input = ComplexTensorHandle::new_contiguous(
        vec![1, 8],
        client.create_from_slice(f32::as_bytes(&values)),
        dtype,
    )
    .unwrap();
    let spectrum = cfft_interleaved(input, 1, FftMode::Forward, FftNormalization::None).unwrap();
    let result = cfft_interleaved(spectrum, 1, FftMode::Inverse, FftNormalization::ByN).unwrap();
    assert_complex_scalars_approx(&client, result, &values, 1e-4);
}

#[test]
fn cfft_interleaved_supports_axis_zero() {
    round_trip(vec![8, 2], 0, FftNormalization::None);
}

#[test]
fn cfft_interleaved_supports_middle_axis() {
    round_trip(vec![2, 8, 3], 1, FftNormalization::None);
}

#[test]
fn cfft_interleaved_supports_batched_windows() {
    round_trip(vec![3, 8], 1, FftNormalization::None);
}

#[test]
fn cfft_interleaved_preserves_scalar_strided_logical_layout() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let shape = vec![2, 8, 3];
    let strides = vec![30, 3, 1];
    let values = values_for(&shape);
    let mut physical = vec![0.0; 108];
    for logical in 0..shape.iter().product::<usize>() {
        let c0 = logical / (shape[1] * shape[2]);
        let c1 = (logical / shape[2]) % shape[1];
        let c2 = logical % shape[2];
        let scalar = 2 * (c0 * strides[0] + c1 * strides[1] + c2 * strides[2]);
        physical[scalar] = values[2 * logical];
        physical[scalar + 1] = values[2 * logical + 1];
    }
    let input = ComplexTensorHandle::new_strided(
        shape,
        strides,
        client.create_from_slice(f32::as_bytes(&physical)),
        dtype,
    )
    .unwrap();
    let spectrum = cfft_interleaved(input, 1, FftMode::Forward, FftNormalization::None).unwrap();
    let result = cfft_interleaved(spectrum, 1, FftMode::Inverse, FftNormalization::ByN).unwrap();
    assert_logical_c32_approx(&client, result, &values, 1e-4);
}

#[test]
fn cfft_interleaved_supports_minimum_n_fft() {
    round_trip(vec![1, 2], 1, FftNormalization::None);
}

#[test]
fn cfft_interleaved_ortho_round_trip() {
    round_trip(vec![1, 8], 1, FftNormalization::Ortho);
}

#[test]
fn cfft_interleaved_shared_memory_boundary_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = test_max_shared_fft_n(&client);
    run_round_trip(&client, vec![1, n_fft, 1], 1, FftNormalization::ByN, 0.03);
}

#[test]
#[cfg(feature = "heavy")]
fn cfft_interleaved_first_four_step_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_four_step_n(&client);
    run_round_trip(&client, vec![1, n_fft, 1], 1, FftNormalization::ByN, 0.03);
}

#[test]
fn cfft_interleaved_rejects_invalid_axis() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let input = contiguous_c32(&client, vec![8], &values_for(&[8]));
    assert!(matches!(
        cfft_interleaved(input, 1, FftMode::Forward, FftNormalization::None),
        Err(FftError::AxisOutOfBounds { dim: 1, rank: 1 })
    ));
}

#[test]
fn cfft_interleaved_rejects_invalid_length() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let input = contiguous_c32(&client, vec![3], &values_for(&[3]));
    assert!(matches!(
        cfft_interleaved(input, 0, FftMode::Forward, FftNormalization::None),
        Err(FftError::InvalidFftLength { n_fft: 3 })
    ));
}

#[test]
fn cfft_interleaved_launch_rejects_shape_mismatch() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let input = contiguous_c32(&client, vec![1, 8], &values_for(&[1, 8]));
    let output = contiguous_c32(&client, vec![2, 4], &values_for(&[2, 4]));
    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            input.binding(),
            output.binding(),
            1,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::ShapeMismatch { name: "output", .. })
    ));
}

#[test]
fn cfft_interleaved_launch_rejects_aliased_output() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let handle = client.create_from_slice(f32::as_bytes(&values_for(&[8])));
    let input = ComplexTensorHandle::new_contiguous(vec![8], handle.clone(), dtype).unwrap();
    let output = ComplexTensorHandle::new_contiguous(vec![8], handle, dtype).unwrap();
    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            input.binding(),
            output.binding(),
            0,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::OverlappingBindings)
    ));
}

#[test]
fn cfft_interleaved_launch_rejects_same_tensor_as_input_and_output() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tensor = contiguous_c32(&client, vec![8], &values_for(&[8]));
    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            tensor.binding(),
            tensor.binding(),
            0,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::OverlappingBindings)
    ));
}

#[test]
fn cfft_interleaved_launch_rejects_overlapping_output_layout() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let input = contiguous_c32(&client, vec![2, 2], &values_for(&[2, 2]));
    let output = ComplexTensorHandle::new_strided(
        vec![2, 2],
        vec![0, 1],
        client.empty(4 * dtype.size()),
        dtype,
    )
    .unwrap();
    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            input.binding(),
            output.binding(),
            1,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::OverlappingBindings)
    ));
}

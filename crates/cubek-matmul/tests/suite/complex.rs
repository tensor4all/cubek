use cubecl::{
    Runtime, TestRuntime,
    client::ComputeClient,
    ir::StorageType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::{ComplexMatmulOptions, Strategy, launch_c32_ref},
};
use cubek_std::InputBinding;
use num_complex::Complex32;

const EPS: f32 = 1.0e-4;

#[test]
fn c32_rank2_without_conjugation_matches_reference() {
    run_case(
        &[2, 3],
        &[3, 2],
        &[2, 2],
        dense_strides(&[2, 3]),
        dense_strides(&[3, 2]),
        dense_strides(&[2, 2]),
        false,
        false,
    );
}

#[test]
fn c32_rank2_with_lhs_conjugation_matches_reference() {
    run_case(
        &[2, 3],
        &[3, 2],
        &[2, 2],
        dense_strides(&[2, 3]),
        dense_strides(&[3, 2]),
        dense_strides(&[2, 2]),
        true,
        false,
    );
}

#[test]
fn c32_rank2_with_rhs_conjugation_matches_reference() {
    run_case(
        &[2, 3],
        &[3, 2],
        &[2, 2],
        dense_strides(&[2, 3]),
        dense_strides(&[3, 2]),
        dense_strides(&[2, 2]),
        false,
        true,
    );
}

#[test]
fn c32_rank2_with_both_conjugated_matches_reference() {
    run_case(
        &[2, 3],
        &[3, 2],
        &[2, 2],
        dense_strides(&[2, 3]),
        dense_strides(&[3, 2]),
        dense_strides(&[2, 2]),
        true,
        true,
    );
}

#[test]
fn c32_batched_with_broadcast_lhs_matches_reference() {
    run_case(
        &[1, 2, 3],
        &[2, 3, 2],
        &[2, 2, 2],
        Strides::new(&[0, 1, 2]),
        dense_strides(&[2, 3, 2]),
        dense_strides(&[2, 2, 2]),
        true,
        true,
    );
}

#[test]
fn c32_strided_bindings_match_reference() {
    run_case(
        &[2, 3],
        &[3, 2],
        &[2, 2],
        Strides::new(&[1, 4]),
        Strides::new(&[1, 5]),
        Strides::new(&[1, 3]),
        false,
        true,
    );
}

fn run_case(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    lhs_strides: Strides,
    rhs_strides: Strides,
    out_strides: Strides,
    lhs_conj: bool,
    rhs_conj: bool,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lhs_values = logical_values(lhs_shape, 0.25);
    let rhs_values = logical_values(rhs_shape, -0.5);
    let expected = reference_complex_matmul(
        lhs_shape,
        rhs_shape,
        out_shape,
        &lhs_values,
        &rhs_values,
        lhs_conj,
        rhs_conj,
    );

    let lhs = c32_handle(&client, lhs_shape, &lhs_strides, &lhs_values);
    let rhs = c32_handle(&client, rhs_shape, &rhs_strides, &rhs_values);
    let out = c32_zeros(&client, out_shape, &out_strides);

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: c32_dtype(),
        rhs: c32_dtype(),
        out: c32_dtype(),
    });

    launch_c32_ref(
        &Strategy::Naive,
        &client,
        InputBinding::new(lhs.binding(), c32_dtype()),
        InputBinding::new(rhs.binding(), c32_dtype()),
        out.clone().binding(),
        &mut dtypes,
        ComplexMatmulOptions { lhs_conj, rhs_conj },
    )
    .unwrap();

    let actual = read_c32_logical(&client, out, out_shape, &out_strides);
    assert_complex_close(&actual, &expected);
}

fn c32_dtype() -> StorageType {
    Complex32::as_type_native_unchecked().storage_type()
}

fn dense_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    Strides::new(&strides)
}

fn physical_extent(shape: &[usize], strides: &Strides) -> usize {
    shape
        .iter()
        .zip(strides.iter())
        .map(|(dim, stride)| if *dim == 0 { 0 } else { (dim - 1) * stride })
        .sum::<usize>()
        + 1
}

fn logical_values(shape: &[usize], seed: f32) -> Vec<Complex32> {
    let len = shape.iter().product();
    (0..len)
        .map(|i| {
            let x = seed + i as f32 * 0.125;
            Complex32::new(x, -0.75 * x + 0.5)
        })
        .collect()
}

fn c32_handle(
    client: &ComputeClient<TestRuntime>,
    shape: &[usize],
    strides: &Strides,
    values: &[Complex32],
) -> TensorHandle<TestRuntime> {
    let mut physical = vec![Complex32::new(0.0, 0.0); physical_extent(shape, strides)];
    for (logical, &value) in values.iter().enumerate() {
        let coord = unravel_row_major(logical, shape);
        let complex_offset = offset(&coord, strides);
        physical[complex_offset] = value;
    }
    let handle = client.create_from_slice(Complex32::as_bytes(&physical));
    TensorHandle::new(
        handle,
        Shape::from(shape.to_vec()),
        strides.clone(),
        c32_dtype(),
    )
}

fn c32_zeros(
    client: &ComputeClient<TestRuntime>,
    shape: &[usize],
    strides: &Strides,
) -> TensorHandle<TestRuntime> {
    let physical = vec![Complex32::new(0.0, 0.0); physical_extent(shape, strides)];
    let handle = client.create_from_slice(Complex32::as_bytes(&physical));
    TensorHandle::new(
        handle,
        Shape::from(shape.to_vec()),
        strides.clone(),
        c32_dtype(),
    )
}

fn read_c32_logical(
    client: &ComputeClient<TestRuntime>,
    handle: TensorHandle<TestRuntime>,
    shape: &[usize],
    strides: &Strides,
) -> Vec<Complex32> {
    let bytes = client.read_one_unchecked(handle.handle);
    let physical = Complex32::from_bytes(&bytes);
    (0..shape.iter().product())
        .map(|logical| {
            let coord = unravel_row_major(logical, shape);
            let complex_offset = offset(&coord, strides);
            physical[complex_offset]
        })
        .collect()
}

fn reference_complex_matmul(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    lhs: &[Complex32],
    rhs: &[Complex32],
    lhs_conj: bool,
    rhs_conj: bool,
) -> Vec<Complex32> {
    let rank = out_shape.len();
    let batch_rank = rank - 2;
    let m = out_shape[rank - 2];
    let n = out_shape[rank - 1];
    let k = lhs_shape[lhs_shape.len() - 1];
    let mut out = vec![Complex32::new(0.0, 0.0); out_shape.iter().product()];

    for out_linear in 0..out.len() {
        let out_coord = unravel_row_major(out_linear, out_shape);
        let row = out_coord[batch_rank];
        let col = out_coord[batch_rank + 1];
        let mut acc = Complex32::new(0.0, 0.0);
        for kk in 0..k {
            let lhs_batch = broadcast_coord(&out_coord[..batch_rank], &lhs_shape[..batch_rank]);
            let rhs_batch = broadcast_coord(&out_coord[..batch_rank], &rhs_shape[..batch_rank]);
            let mut lhs_coord = lhs_batch;
            lhs_coord.push(row);
            lhs_coord.push(kk);
            let mut rhs_coord = rhs_batch;
            rhs_coord.push(kk);
            rhs_coord.push(col);
            let a = maybe_conj(lhs[linear_row_major(&lhs_coord, lhs_shape)], lhs_conj);
            let b = maybe_conj(rhs[linear_row_major(&rhs_coord, rhs_shape)], rhs_conj);
            acc += a * b;
        }
        out[out_linear] = acc;
    }
    debug_assert_eq!(
        out.len(),
        m * n * out_shape[..batch_rank].iter().product::<usize>()
    );
    out
}

fn maybe_conj(value: Complex32, conj: bool) -> Complex32 {
    if conj {
        Complex32::new(value.re, -value.im)
    } else {
        value
    }
}

fn broadcast_coord(out_batch: &[usize], input_batch_shape: &[usize]) -> Vec<usize> {
    let pad = out_batch.len() - input_batch_shape.len();
    input_batch_shape
        .iter()
        .enumerate()
        .map(|(i, dim)| if *dim == 1 { 0 } else { out_batch[pad + i] })
        .collect()
}

fn unravel_row_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coord = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coord[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coord
}

fn linear_row_major(coord: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1;
    let mut linear = 0;
    for axis in (0..shape.len()).rev() {
        linear += coord[axis] * stride;
        stride *= shape[axis];
    }
    linear
}

fn offset(coord: &[usize], strides: &Strides) -> usize {
    coord
        .iter()
        .zip(strides.iter())
        .map(|(idx, stride)| idx * stride)
        .sum()
}

fn assert_complex_close(actual: &[Complex32], expected: &[Complex32]) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a.re - e.re).abs().max((a.im - e.im).abs()))
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= EPS,
        "max complex GEMM error {max_error} exceeds {EPS}\nactual={actual:?}\nexpected={expected:?}",
    );
}

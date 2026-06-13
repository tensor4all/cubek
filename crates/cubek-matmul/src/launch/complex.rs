use core::marker::PhantomData;

use cubecl::{
    Runtime, calculate_cube_count_elemwise,
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    zspace::{Shape, Strides},
};
use cubek_std::{InputBinding, MatrixLayout};
use num_complex::Complex32;

use crate::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
    launch::{Strategy, launch_ref},
};

type C32Parts = Vector<f32, Const<2>>;

/// Options for CubeK-owned `C32` matrix multiplication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComplexMatmulOptions {
    /// Conjugate the left-hand-side complex input before multiplication.
    pub lhs_conj: bool,
    /// Conjugate the right-hand-side complex input before multiplication.
    pub rhs_conj: bool,
}

#[derive(Clone, Copy)]
enum C32Part {
    Real,
    Imag,
}

/// Launch a `C32` matmul by lowering to four real `F32` matmuls.
#[allow(clippy::result_large_err)]
pub fn launch_c32_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
    options: ComplexMatmulOptions,
) -> Result<(), MatmulSetupError> {
    validate_c32_globals(dtypes)?;
    validate_normal_input("lhs", &lhs)?;
    validate_normal_input("rhs", &rhs)?;
    validate_rank("lhs", lhs.shape())?;
    validate_rank("rhs", rhs.shape())?;
    validate_rank("out", &out.shape)?;

    let lhs_real = extract_part(client, lhs.data(), C32Part::Real, MatrixLayout::RowMajor)?;
    let lhs_imag = extract_part(client, lhs.data(), C32Part::Imag, MatrixLayout::RowMajor)?;
    let rhs_real = extract_part(client, rhs.data(), C32Part::Real, MatrixLayout::ColMajor)?;
    let rhs_imag = extract_part(client, rhs.data(), C32Part::Imag, MatrixLayout::ColMajor)?;

    let out_shape = out.shape.clone();
    let out_strides = dense_matrix_strides(out_shape.as_slice(), MatrixLayout::RowMajor)?;
    let real_pos = scratch_like::<R>(client, &out_shape, &out_strides)?;
    let real_neg = scratch_like::<R>(client, &out_shape, &out_strides)?;
    let imag_left = scratch_like::<R>(client, &out_shape, &out_strides)?;
    let imag_right = scratch_like::<R>(client, &out_shape, &out_strides)?;

    let f32_type = f32_storage_type();
    let mut real_dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f32_type,
        rhs: f32_type,
        out: f32_type,
    });

    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_real.clone(), f32_type),
        InputBinding::new(rhs_real.clone(), f32_type),
        real_pos.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_imag.clone(), f32_type),
        InputBinding::new(rhs_imag.clone(), f32_type),
        real_neg.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_real, f32_type),
        InputBinding::new(rhs_imag, f32_type),
        imag_left.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_imag, f32_type),
        InputBinding::new(rhs_real, f32_type),
        imag_right.clone(),
        &mut real_dtypes,
    )?;

    *dtypes = real_dtypes;
    compose_parts(
        client,
        &out,
        &real_pos,
        &real_neg,
        &imag_left,
        &imag_right,
        options,
    )
}

#[cube(launch_unchecked)]
fn extract_c32_part_kernel(
    input_meta: &Tensor<f32>,
    input_parts: &Array<C32Parts>,
    out_parts: &mut Tensor<f32>,
    #[comptime] part: u32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= out_parts.len() {
        terminate!();
    }

    let mut input_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = out_parts.coordinate(ABSOLUTE_POS, axis);
        input_offset += coord * input_meta.stride(axis);
    }

    let value = input_parts[input_offset];
    out_parts[ABSOLUTE_POS] = if part == 0 { value[0] } else { value[1] };
}

#[cube(launch_unchecked)]
fn compose_c32_parts_kernel(
    out_meta: &Tensor<f32>,
    out_parts: &mut Array<C32Parts>,
    real_pos: &Tensor<f32>,
    real_neg: &Tensor<f32>,
    imag_left: &Tensor<f32>,
    imag_right: &Tensor<f32>,
    #[comptime] lhs_imag_sign: i32,
    #[comptime] rhs_imag_sign: i32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= real_pos.len() {
        terminate!();
    }

    let mut out_offset = 0usize;
    let mut dense_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = real_pos.coordinate(ABSOLUTE_POS, axis);
        out_offset += coord * out_meta.stride(axis);
        dense_offset += coord * real_pos.stride(axis);
    }

    let lhs_sign = f32::cast_from(lhs_imag_sign);
    let rhs_sign = f32::cast_from(rhs_imag_sign);
    let real = real_pos[dense_offset] - lhs_sign * rhs_sign * real_neg[dense_offset];
    let imag = rhs_sign * imag_left[dense_offset] + lhs_sign * imag_right[dense_offset];
    let mut value = Vector::<f32, Const<2>>::empty();
    value[0] = real;
    value[1] = imag;
    out_parts[out_offset] = value;
}

fn extract_part<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorBinding<R>,
    part: C32Part,
    layout: MatrixLayout,
) -> Result<TensorBinding<R>, MatmulSetupError> {
    let strides = dense_matrix_strides(input.shape.as_slice(), layout)?;
    let out = scratch_like(client, &input.shape, &strides)?;
    let len = logical_len(input.shape.as_slice())?;
    let cube_dim = CubeDim::new(client, len);
    let cube_count = calculate_cube_count_elemwise(client, len, cube_dim);
    let part_value = match part {
        C32Part::Real => 0,
        C32Part::Imag => 1,
    };
    let input_parts = c32_array_arg(input);

    // SAFETY: The launch covers the dense f32 scratch output domain. Logical
    // input indexing uses TensorBinding shape/stride metadata and never turns
    // tensor extents or strides into compile-time constants.
    unsafe {
        extract_c32_part_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            input.clone().into_tensor_arg(),
            input_parts,
            out.clone().into_tensor_arg(),
            part_value,
            input.shape.len(),
        )
    };
    Ok(out)
}

fn compose_parts<R: Runtime>(
    client: &ComputeClient<R>,
    out: &TensorBinding<R>,
    real_pos: &TensorBinding<R>,
    real_neg: &TensorBinding<R>,
    imag_left: &TensorBinding<R>,
    imag_right: &TensorBinding<R>,
    options: ComplexMatmulOptions,
) -> Result<(), MatmulSetupError> {
    let len = logical_len(out.shape.as_slice())?;
    let cube_dim = CubeDim::new(client, len);
    let cube_count = calculate_cube_count_elemwise(client, len, cube_dim);
    let lhs_sign = if options.lhs_conj { -1 } else { 1 };
    let rhs_sign = if options.rhs_conj { -1 } else { 1 };
    let out_meta = metadata_with_backing(out, real_pos);
    let out_parts = c32_array_arg(out);

    // SAFETY: Product tensors are dense f32 scratch tensors with the same
    // logical shape as the C32 output. Output writes use a raw f32x2 array;
    // output shape/stride metadata is carried separately at runtime.
    unsafe {
        compose_c32_parts_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            out_meta.into_tensor_arg(),
            out_parts,
            real_pos.clone().into_tensor_arg(),
            real_neg.clone().into_tensor_arg(),
            imag_left.clone().into_tensor_arg(),
            imag_right.clone().into_tensor_arg(),
            lhs_sign,
            rhs_sign,
            out.shape.len(),
        )
    };
    Ok(())
}

fn validate_normal_input<R: Runtime>(
    name: &'static str,
    input: &InputBinding<R>,
) -> Result<(), MatmulSetupError> {
    match input {
        InputBinding::Normal(_, dtype) if *dtype == c32_storage_type() => Ok(()),
        InputBinding::Normal(_, dtype) => Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} must use C32 storage, got {dtype:?}"
        )))),
        InputBinding::Quantized { .. } => Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} does not support quantized input"
        )))),
    }
}

fn validate_c32_globals(dtypes: &MatmulElems) -> Result<(), MatmulSetupError> {
    let c32_type = c32_storage_type();
    if dtypes.lhs_global != c32_type
        || dtypes.rhs_global != c32_type
        || dtypes.acc_global != c32_type
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM dtypes must be C32, got lhs={:?}, rhs={:?}, out={:?}",
            dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global
        ))));
    }
    Ok(())
}

fn validate_rank(name: &'static str, shape: &Shape) -> Result<(), MatmulSetupError> {
    if shape.len() < 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} must have rank at least 2",
        ))));
    }
    Ok(())
}

fn scratch_like<R: Runtime>(
    client: &ComputeClient<R>,
    shape: &Shape,
    strides: &Strides,
) -> Result<TensorBinding<R>, MatmulSetupError> {
    let len = logical_len(shape.as_slice())?;
    let handle = client.empty(len * f32_storage_type().size()).binding();
    Ok(TensorBinding {
        handle,
        strides: strides.clone(),
        shape: shape.clone(),
        runtime: PhantomData,
    })
}

fn dense_matrix_strides(
    shape: &[usize],
    layout: MatrixLayout,
) -> Result<Strides, MatmulSetupError> {
    let rank = shape.len();
    let mut strides = vec![0usize; rank];
    let rows = shape[rank - 2];
    let cols = shape[rank - 1];
    match layout {
        MatrixLayout::RowMajor => {
            strides[rank - 2] = cols;
            strides[rank - 1] = 1;
        }
        MatrixLayout::ColMajor => {
            strides[rank - 2] = 1;
            strides[rank - 1] = rows;
        }
    }

    let mut batch_stride = rows.checked_mul(cols).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new("complex GEMM batch stride overflow"))
    })?;
    for axis in (0..rank - 2).rev() {
        strides[axis] = batch_stride;
        batch_stride = batch_stride.checked_mul(shape[axis]).ok_or_else(|| {
            MatmulSetupError::InvalidConfig(Box::new("complex GEMM batch stride overflow"))
        })?;
    }
    Ok(Strides::new(&strides))
}

fn c32_array_arg<R: Runtime>(binding: &TensorBinding<R>) -> ArrayArg<R> {
    let len = binding.handle.size() as usize / c32_storage_type().size();
    // SAFETY: The length is derived from the actual C32 binding size in complex
    // elements, so strided logical accesses can reach padded physical offsets.
    unsafe { ArrayArg::from_raw_parts_binding(binding.handle.clone(), len) }
}

fn metadata_with_backing<R: Runtime>(
    metadata: &TensorBinding<R>,
    backing: &TensorBinding<R>,
) -> TensorBinding<R> {
    TensorBinding {
        handle: backing.handle.clone(),
        strides: metadata.strides.clone(),
        shape: metadata.shape.clone(),
        runtime: PhantomData,
    }
}

fn logical_len(shape: &[usize]) -> Result<usize, MatmulSetupError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            MatmulSetupError::InvalidConfig(Box::new("complex GEMM logical length overflow"))
        })
    })
}

fn f32_storage_type() -> StorageType {
    StorageType::Scalar(ElemType::Float(FloatKind::F32))
}

fn c32_storage_type() -> StorageType {
    Complex32::as_type_native_unchecked().storage_type()
}

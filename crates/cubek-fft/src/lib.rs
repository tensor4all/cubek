//! FFT primitives for CubeCL runtimes.
//!
//! # Interleaved C32 ABI
//!
//! The interleaved entry points accept [`ComplexTensorHandle`] values with one
//! logical C32 element per complex value. Physically, each logical element is
//! stored as two adjacent F32 scalars in `[re, im]` order. Shapes and the
//! strides accepted by [`ComplexTensorHandle::new_strided`] are therefore in
//! logical complex-element units; [`ComplexTensorHandle::scalar_strides`] is
//! the corresponding physical F32-scalar stride (`2 * logical_stride`).
//! `num_complex::Complex32` is `#[repr(C)]` with adjacent `re: f32` and
//! `im: f32` fields, so a contiguous slice has the same bytes as this ABI
//! when its starting address satisfies `align_of::<Complex32>()` (currently
//! the same four-byte alignment enforced for F32 offsets). This describes
//! layout compatibility only; callers must still use a sound ownership-aware
//! cast or copy when moving between Rust slices and runtime buffers.
//!
//! RFFT inputs and IRFFT outputs retain their ordinary real logical shape.
//! The interleaved RFFT output and IRFFT input use the same logical shape with
//! the transformed axis shortened to `n_fft / 2 + 1`. Caller-provided outputs
//! must be unique, non-overlapping allocations; aliased bindings and
//! overlapping output layouts are rejected.
//!
//! | [`FftNormalization`] | Scale applied to every transform direction |
//! | --- | --- |
//! | `None` | `1` |
//! | `ByN` | `1 / n_fft` |
//! | `Ortho` | `1 / sqrt(n_fft)` |
//!
//! Interleaved CFFT chooses the shared-memory small FFT path up to the active
//! device's shared-size limit and the four-step path above it. The allocating
//! helpers use the runtime's default device; use the `*_launch` functions with
//! an explicit `ComputeClient` to select a device. This release supports only
//! F32 real tensors and C32 interleaved tensors. F64/C64 support is deferred.
//!
//! Profiling an interleaved launch should show no standalone interleaved
//! conversion or scaling kernel. Large real transforms still use their
//! algorithmic real-to-packed-CFFT and packed-CFFT-to-real stages internally;
//! their external complex boundary reads or writes `[re, im]` directly and
//! final scaling is fused into the existing store.

mod complex;
mod error;
mod fft;
mod interleaved_layout;
mod layout;
mod normalization;

pub use complex::{ComplexTensorBinding, ComplexTensorHandle};
pub use error::FftError;
pub use fft::*;
pub use normalization::FftNormalization;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;

# Interleaved C32 FFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add public, panic-free interleaved F32/C32 CFFT, RFFT, and IRFFT APIs to `cubek-fft`, covering small and four-step paths without standalone pack/unpack kernels.

**Architecture:** `ComplexTensorHandle` stores one scalar allocation with logical complex shape and scalar-space strides; `ComplexTensorBinding` borrows it for launches. Component-aware layouts map each logical complex coordinate to adjacent even/odd F32 positions. Global I/O is interleaved, while shared memory and internal four-step/packed scratch remain split.

**Tech Stack:** Rust 2024, CubeCL/CubeK, `thiserror`, `num-complex` CPU references, Cargo integration tests, WGPU/Metal test runtime.

## Global Constraints

- Target `tensor4all/cubek:main`; reference issue #6 without closing it.
- Existing split APIs, signatures, numerical behavior, and normalization behavior remain unchanged.
- The first pull request supports F32/C32 only; F64/C64 execution is a follow-up pull request.
- C32 physical order is `[re0, im0, re1, im1, ...]`; no visible trailing dimension of length two.
- Reuse `max_shared_fft_n(client)` for small/four-step selection; do not add a fixed 4096 threshold.
- No standalone pack/unpack or scale kernel, no host staging, and no CPU fallback. Caller-owned launch APIs perform no hidden output allocation; allocating convenience APIs allocate only their documented output.
- Internal split shared memory and split four-step/packed scratch remain unchanged.
- New public APIs return `Result<_, FftError>` and do not validate user input with `assert!` or `unwrap`.
- Out-of-place launches reject aliased writable output before the first launch or scratch allocation.
- Run `codegraph sync` after structural changes so the repository index remains current locally; never commit `.codegraph/`.

## File map

- Create `crates/cubek-fft/src/complex.rs`: complex handle/binding, physical extent validation, output uniqueness check.
- Create `crates/cubek-fft/src/error.rs`: public typed FFT validation/launch errors.
- Create `crates/cubek-fft/src/normalization.rs`: public normalization enum and scale helpers.
- Create `crates/cubek-fft/src/interleaved_layout.rs`: CubeCL component-aware real/imaginary layouts.
- Create `crates/cubek-fft/src/fft/cfft_interleaved.rs`: public CFFT API plus small/four-step interleaved kernels.
- Create `crates/cubek-fft/src/fft/rfft_interleaved.rs`: public RFFT API plus small interleaved output path.
- Create `crates/cubek-fft/src/fft/irfft_interleaved.rs`: public IRFFT API plus small interleaved input path.
- Modify `crates/cubek-fft/src/fft/rfft_large.rs`: add interleaved large RFFT/IRFFT boundary kernels while retaining split scratch.
- Modify `crates/cubek-fft/src/lib.rs` and `crates/cubek-fft/src/fft/mod.rs`: export the additive API.
- Modify `crates/cubek-fft/Cargo.toml`: add the workspace `thiserror` dependency.
- Create `crates/cubek-fft/tests/fft/interleaved_cfft.rs`, `interleaved_rfft.rs`, `interleaved_irfft.rs`, and `interleaved_validation.rs`.
- Modify `crates/cubek-fft/tests/fft/mod.rs`: register the new tests.
- Modify `crates/cubek-fft/src/eval/benchmarks/{strategy.rs,benchmark.rs,problem.rs}` and FFT documentation for interleaved benchmark coverage.

---

### Task 1: Complex tensor ABI, errors, and normalization

**Files:**
- Create: `crates/cubek-fft/src/complex.rs`
- Create: `crates/cubek-fft/src/error.rs`
- Create: `crates/cubek-fft/src/normalization.rs`
- Create: `crates/cubek-fft/tests/fft/interleaved_validation.rs`
- Modify: `crates/cubek-fft/src/lib.rs`
- Modify: `crates/cubek-fft/Cargo.toml`
- Modify: `crates/cubek-fft/tests/fft/mod.rs`

**Interfaces:**
- Produces `ComplexTensorHandle<R>`, `ComplexTensorBinding<'a, R>`, `FftError`, and `FftNormalization` for all later tasks.
- `ComplexTensorHandle` stores a `TensorHandle<R>` whose shape is logical and whose strides are physical scalar strides.
- Public constructors accept logical complex strides and multiply them by two with checked arithmetic.

- [ ] **Step 1: Write failing ABI and normalization tests**

Add these tests to `tests/fft/interleaved_validation.rs`:

```rust
use cubecl::{CubeElement, Runtime, TestRuntime, frontend::CubePrimitive};
use cubek_fft::{ComplexTensorHandle, FftError, FftNormalization};

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
        vec![4], client.empty(8 * f64_dtype.size()), f64_dtype,
    );
    assert!(matches!(wrong, Err(FftError::UnsupportedDtype { .. })));
    let short = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        vec![4], client.empty(7 * f32_dtype.size()), f32_dtype,
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
        vec![2], vec![], client.empty(8 * dtype.size()), dtype,
    );
    assert!(matches!(rank, Err(FftError::RankMismatch { .. })));

    let misaligned = ComplexTensorHandle::<TestRuntime>::new_contiguous(
        vec![2], client.empty(5 * dtype.size()).offset_start(1), dtype,
    );
    assert!(matches!(misaligned, Err(FftError::MisalignedBuffer { .. })));

    let stride_overflow = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![2], vec![usize::MAX], client.empty(4 * dtype.size()), dtype,
    );
    assert!(matches!(stride_overflow, Err(FftError::StrideOverflow { axis: 0 })));

    let extent_overflow = ComplexTensorHandle::<TestRuntime>::new_strided(
        vec![usize::MAX, 2], vec![1, 1], client.empty(4 * dtype.size()), dtype,
    );
    assert!(matches!(extent_overflow, Err(FftError::SizeOverflow)));
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p cubek-fft --test lib interleaved_validation -- --nocapture
```

Expected: compilation fails because `ComplexTensorHandle`, `FftError`, and `FftNormalization` do not exist.

- [ ] **Step 3: Implement the public host-side types**

Add `thiserror = { workspace = true }` to `cubek-fft` dependencies. Implement these exact public surfaces:

```rust
#[derive(Debug, thiserror::Error)]
pub enum FftError {
    #[error("unsupported FFT storage dtype {actual:?}; expected F32")]
    UnsupportedDtype { actual: StorageType },
    #[error("shape rank {shape_rank} differs from stride rank {stride_rank}")]
    RankMismatch { shape_rank: usize, stride_rank: usize },
    #[error("FFT axis {dim} is out of bounds for rank {rank}")]
    AxisOutOfBounds { dim: usize, rank: usize },
    #[error("FFT length must be a power of two and at least 2, got {n_fft}")]
    InvalidFftLength { n_fft: usize },
    #[error("{name}={value} is outside {min}..={max}")]
    InvalidLength { name: &'static str, value: usize, min: usize, max: usize },
    #[error("complex buffer needs {required} scalar elements but only {available} are available")]
    InsufficientBuffer { required: usize, available: usize },
    #[error("complex buffer byte offset {offset} is not aligned to scalar size {scalar_size}")]
    MisalignedBuffer { offset: u64, scalar_size: usize },
    #[error("complex scalar stride at axis {axis} overflowed")]
    StrideOverflow { axis: usize },
    #[error("complex buffer extent overflowed")]
    SizeOverflow,
    #[error("{name} shape {actual:?} does not match expected shape {expected:?}")]
    ShapeMismatch { name: &'static str, actual: Vec<usize>, expected: Vec<usize> },
    #[error("input and output allocations overlap")]
    OverlappingBindings,
    #[error(transparent)]
    Launch(#[from] cubecl::prelude::LaunchError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftNormalization { None, ByN, Ortho }

impl FftNormalization {
    pub fn scale_f32(self, n_fft: usize) -> Result<f32, FftError> {
        if n_fft < 2 || !n_fft.is_power_of_two() {
            return Err(FftError::InvalidFftLength { n_fft });
        }
        Ok(match self {
            Self::None => 1.0,
            Self::ByN => 1.0 / n_fft as f32,
            Self::Ortho => 1.0 / (n_fft as f32).sqrt(),
        })
    }
}
```

Implement `ComplexTensorHandle::empty`, `new_contiguous`, `new_strided`, `shape`, `strides`, `scalar_strides`, `physical_scalar_len`, `dtype`, `binding`, and `into_raw_parts`. `ComplexTensorBinding<'a, R>` borrows the handle and exposes crate-private `tensor()` for launch code. Use checked multiplication/addition for scalar strides and extent.

Before production code, extend the failing test file with zero-sized shape and a successful non-contiguous extent case. Assert `physical_scalar_len() == 0` for the former and the checked maximum reachable imaginary scalar plus one for the latter.

Implement crate-private output validation with the CubeCL handle-count contract:

```rust
pub(crate) fn ensure_unique_output<R: Runtime>(tensor: &TensorHandle<R>) -> Result<(), FftError> {
    let probe = tensor.handle.clone();
    if probe.can_mut() { Ok(()) } else { Err(FftError::OverlappingBindings) }
}
```

The probe makes a unique output have two handles (accepted) and an aliased input/output have at least three (rejected). Perform this check before cloning a handle into a CubeCL `TensorBinding`.

- [ ] **Step 4: Run focused and crate tests and verify GREEN**

```bash
cargo test -p cubek-fft --test lib interleaved_validation -- --nocapture
cargo test -p cubek-fft
cargo fmt --all -- --check
```

Expected: the new focused tests pass; all existing 14 FFT integration tests still pass; formatting is clean.

- [ ] **Step 5: Commit the host-side ABI**

```bash
git add crates/cubek-fft/Cargo.toml crates/cubek-fft/src/lib.rs \
  crates/cubek-fft/src/complex.rs crates/cubek-fft/src/error.rs \
  crates/cubek-fft/src/normalization.rs crates/cubek-fft/tests/fft/mod.rs \
  crates/cubek-fft/tests/fft/interleaved_validation.rs
git commit -m "feat(fft): add interleaved complex tensor ABI"
```

---

### Task 2: Component layout and small interleaved CFFT

**Files:**
- Create: `crates/cubek-fft/src/interleaved_layout.rs`
- Create: `crates/cubek-fft/src/fft/cfft_interleaved.rs`
- Create: `crates/cubek-fft/tests/fft/interleaved_cfft.rs`
- Modify: `crates/cubek-fft/src/lib.rs`
- Modify: `crates/cubek-fft/src/fft/mod.rs`
- Modify: `crates/cubek-fft/tests/fft/mod.rs`

**Interfaces:**
- Consumes Task 1 types.
- Produces `cfft_interleaved`, `cfft_interleaved_launch`, and the component layout used by RFFT/IRFFT.

- [ ] **Step 1: Write a failing small CFFT round-trip test**

Create an 8-element C32 input with known scalar order, launch forward `None`, then inverse `ByN`, and compare the returned raw scalar buffer to the input:

```rust
#[test]
fn cfft_interleaved_small_round_trip_preserves_c32_order() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let values: Vec<f32> = (0..8).flat_map(|i| [i as f32 + 0.25, -(i as f32)]).collect();
    let input = ComplexTensorHandle::new_contiguous(
        vec![1, 8], client.create_from_slice(f32::as_bytes(&values)), dtype,
    ).unwrap();
    let spectrum = cfft_interleaved(input, 1, FftMode::Forward, FftNormalization::None).unwrap();
    let result = cfft_interleaved(spectrum, 1, FftMode::Inverse, FftNormalization::ByN).unwrap();
    assert_complex_scalars_approx(&client, result, &values, 1e-4);
}
```

Also add axis-0, middle-axis, batched, scalar-strided logical layout, minimum `n_fft = 2`, `Ortho` round-trip, invalid-axis, invalid-length, shape mismatch, and aliased-output tests using a shared helper in the same file.

- [ ] **Step 2: Run the small CFFT test and verify RED**

```bash
cargo test -p cubek-fft --test lib cfft_interleaved_small_round_trip_preserves_c32_order -- --nocapture
```

Expected: compilation fails because `cfft_interleaved` and its launch function do not exist.

- [ ] **Step 3: Implement component views and small CFFT**

Implement `InterleavedBatchSignalLayout` with scalar-space metadata:

```rust
#[derive(CubeType, Clone, Copy)]
pub(crate) struct InterleavedBatchSignalLayout {
    num_samples: usize,
    stride_samples: usize,
    batch_offset: usize,
    component: usize,
}
```

`to_source_pos(coords)` returns `batch_offset + coords * stride_samples + component`; `shape()` returns the logical axis length. Build real and imaginary views with components 0 and 1.

Expose these signatures:

```rust
pub fn cfft_interleaved<R: Runtime>(
    input: ComplexTensorHandle<R>, dim: usize, mode: FftMode,
    normalization: FftNormalization,
) -> Result<ComplexTensorHandle<R>, FftError>;

pub fn cfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>, input: ComplexTensorBinding<'_, R>,
    output: ComplexTensorBinding<'_, R>, dim: usize, mode: FftMode,
    normalization: FftNormalization,
) -> Result<(), FftError>;
```

Validate all metadata and output uniqueness before cloning either tensor handle. For `n_fft <= max_shared_fft_n(client)`, launch a new small kernel that reads component views into existing split shared arrays, calls `fft_butterfly_parallel`, and writes `shared_re[k] * scale` and `shared_im[k] * scale` to component views. Keep the existing split kernel unchanged.

- [ ] **Step 4: Verify small paths and compatibility**

```bash
cargo test -p cubek-fft --test lib interleaved_cfft -- --nocapture
cargo test -p cubek-fft
cargo fmt --all -- --check
```

Expected: all light interleaved CFFT cases and the existing split suite pass.

- [ ] **Step 5: Commit small CFFT**

```bash
git add crates/cubek-fft/src/lib.rs crates/cubek-fft/src/interleaved_layout.rs \
  crates/cubek-fft/src/fft/mod.rs crates/cubek-fft/src/fft/cfft_interleaved.rs \
  crates/cubek-fft/tests/fft/mod.rs crates/cubek-fft/tests/fft/interleaved_cfft.rs
git commit -m "feat(fft): add small interleaved CFFT"
```

---

### Task 3: Four-step interleaved CFFT

**Files:**
- Modify: `crates/cubek-fft/src/fft/cfft.rs`
- Modify: `crates/cubek-fft/src/fft/cfft_interleaved.rs`
- Modify: `crates/cubek-fft/tests/fft/interleaved_cfft.rs`

**Interfaces:**
- Extends Task 2 launch functions without changing signatures.
- Reuses `factor_four_step`, `max_shared_fft_n(client)`, and existing split scratch allocation.

- [ ] **Step 1: Add failing first-four-step and boundary tests**

Add this test-only calculation, which mirrors the launch decision from reported hardware properties without exposing an internal selector:

```rust
fn test_max_shared_fft_n(client: &ComputeClient<TestRuntime>) -> usize {
    let max_elems = client.properties().hardware.max_shared_memory_size
        / (2 * core::mem::size_of::<f32>());
    if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() >> 1
    }
}

fn first_four_step_n(client: &ComputeClient<TestRuntime>) -> usize {
    2 * test_max_shared_fft_n(client)
}
```

Test `n_fft = test_max_shared_fft_n(&client)` and `n_fft = first_four_step_n(&client)`. Mark the four-step numerical case `#[cfg(feature = "heavy")]` when the device-reported limit makes it expensive.

```rust
#[test]
#[cfg(feature = "heavy")]
fn cfft_interleaved_first_four_step_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_four_step_n(&client);
    run_round_trip(&client, vec![1, n_fft, 1], 1, FftNormalization::ByN, 0.03);
}
```

- [ ] **Step 2: Verify the large test fails for the missing dispatch**

```bash
cargo test -p cubek-fft --features heavy --test lib cfft_interleaved_first_four_step_round_trip -- --nocapture
```

Expected: the launch returns a typed unsupported-size/internal-path error or the test fails because only the small path exists.

- [ ] **Step 3: Implement the hybrid four-step path**

For `n_fft > max_shared_fft_n(client)`:

1. Allocate the existing split `scratch_re` and `scratch_im` with logical shape.
2. First radix kernel reads interleaved real/imag component views and writes split scratch with the existing post-twiddle.
3. Change `cfft_four_step_radix2_kernel` in `cfft.rs` from private to `pub(crate)` and call the same generated launch module from the interleaved path. Its split, in-place signature remains unchanged.
4. Final transpose reads split scratch and writes adjacent interleaved output scalars multiplied by the requested scale.

Factor both dimensions with `factor_four_step(n_fft, max_shared_fft_n(client))`; return `FftError::InvalidFftLength`/`SizeOverflow` instead of asserting on user-controlled lengths.

- [ ] **Step 4: Verify small/four-step boundary and numerical tests**

```bash
cargo test -p cubek-fft --features heavy --test lib interleaved_cfft -- --nocapture
cargo test -p cubek-fft --features heavy
cargo fmt --all -- --check
```

Expected: boundary selection, both numerical paths, and existing heavy split tests pass.

- [ ] **Step 5: Commit four-step CFFT**

```bash
git add crates/cubek-fft/src/fft/cfft.rs crates/cubek-fft/src/fft/cfft_interleaved.rs \
  crates/cubek-fft/tests/fft/interleaved_cfft.rs
git commit -m "feat(fft): add four-step interleaved CFFT"
```

---

### Task 4: Small and padded interleaved RFFT

**Files:**
- Create: `crates/cubek-fft/src/fft/rfft_interleaved.rs`
- Create: `crates/cubek-fft/tests/fft/interleaved_rfft.rs`
- Modify: `crates/cubek-fft/src/fft/mod.rs`
- Modify: `crates/cubek-fft/tests/fft/mod.rs`

**Interfaces:**
- Produces allocating, caller-owned, and padded interleaved RFFT functions.

- [ ] **Step 1: Add failing numerical, layout, normalization, and padding tests**

Expose and test:

```rust
pub fn rfft_interleaved<R: Runtime>(
    signal: TensorHandle<R>, dim: usize, normalization: FftNormalization,
) -> Result<ComplexTensorHandle<R>, FftError>;

pub fn rfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>, signal: &TensorHandle<R>,
    spectrum: ComplexTensorBinding<'_, R>, dim: usize,
    normalization: FftNormalization,
) -> Result<(), FftError>;

pub fn rfft_interleaved_launch_padded<R: Runtime>(
    client: &ComputeClient<R>, signal: &TensorHandle<R>,
    spectrum: ComplexTensorBinding<'_, R>, dim: usize, signal_len: usize,
    normalization: FftNormalization,
) -> Result<(), FftError>;
```

Test axis 0/middle/last, trailing batches, direct `[re, im]` output order against `rfft_ref`, all normalization variants, and virtual padding matching materialized zeros.

- [ ] **Step 2: Run a focused test and verify RED**

```bash
cargo test -p cubek-fft --test lib rfft_interleaved_axis_last_matches_reference -- --nocapture
```

Expected: compilation fails because the RFFT interleaved API is missing.

- [ ] **Step 3: Implement small RFFT direct stores**

Copy only the small-path control flow needed from `rfft.rs`. Keep real input/shared arrays/butterfly unchanged. Replace the two split output tensors with one complex binding and two `InterleavedBatchSignalLayout` component views. Multiply both components by `normalization.scale_f32(n_fft)?` in the final store. Validate dtype, shape, axis, `signal_len`, output extent, and output uniqueness before launch.

- [ ] **Step 4: Run RFFT and regression suites**

```bash
cargo test -p cubek-fft --test lib interleaved_rfft -- --nocapture
cargo test -p cubek-fft
cargo fmt --all -- --check
```

Expected: all new light/padding tests and existing split tests pass.

- [ ] **Step 5: Commit interleaved RFFT small path**

```bash
git add crates/cubek-fft/src/fft/mod.rs crates/cubek-fft/src/fft/rfft_interleaved.rs \
  crates/cubek-fft/tests/fft/mod.rs crates/cubek-fft/tests/fft/interleaved_rfft.rs
git commit -m "feat(fft): add small interleaved RFFT"
```

---

### Task 5: Small and padded interleaved IRFFT

**Files:**
- Create: `crates/cubek-fft/src/fft/irfft_interleaved.rs`
- Create: `crates/cubek-fft/tests/fft/interleaved_irfft.rs`
- Modify: `crates/cubek-fft/src/fft/mod.rs`
- Modify: `crates/cubek-fft/tests/fft/mod.rs`

**Interfaces:**
- Produces allocating, caller-owned, and padded interleaved IRFFT functions.

- [ ] **Step 1: Add failing numerical, normalization, and padding tests**

Use these public signatures:

```rust
pub fn irfft_interleaved<R: Runtime>(
    spectrum: ComplexTensorHandle<R>, dim: usize,
    normalization: FftNormalization,
) -> Result<TensorHandle<R>, FftError>;

pub fn irfft_interleaved_launch<R: Runtime>(
    client: &ComputeClient<R>, spectrum: ComplexTensorBinding<'_, R>,
    signal: &TensorHandle<R>, dim: usize, normalization: FftNormalization,
) -> Result<(), FftError>;

pub fn irfft_interleaved_launch_padded<R: Runtime>(
    client: &ComputeClient<R>, spectrum: ComplexTensorBinding<'_, R>,
    signal: &TensorHandle<R>, dim: usize, spec_bins: usize,
    normalization: FftNormalization,
) -> Result<(), FftError>;
```

Test axis 0/middle/last, trailing batches, all normalization variants, DC-only input, and virtual spectrum padding. For normalization expectations, derive `None` and `Ortho` from the existing `ByN` CPU reference by multiplying by `n_fft` and `sqrt(n_fft)` respectively.

- [ ] **Step 2: Run a focused test and verify RED**

```bash
cargo test -p cubek-fft --test lib irfft_interleaved_axis_last_matches_reference -- --nocapture
```

Expected: compilation fails because the IRFFT interleaved API is missing.

- [ ] **Step 3: Implement small IRFFT direct loads**

Copy only the small-path control flow needed from `irfft.rs`. Read the half-spectrum from real and imaginary component views, reconstruct conjugate bins in the existing split shared arrays, run the inverse butterfly, and write the real signal with the selected scale. Do not change the existing split IRFFT's hard-coded `1 / n_fft` behavior. Validate output uniqueness before converting its handle to a CubeCL binding.

- [ ] **Step 4: Run IRFFT and full light suites**

```bash
cargo test -p cubek-fft --test lib interleaved_irfft -- --nocapture
cargo test -p cubek-fft
cargo fmt --all -- --check
```

Expected: new IRFFT tests and all existing split tests pass.

- [ ] **Step 5: Commit interleaved IRFFT small path**

```bash
git add crates/cubek-fft/src/fft/mod.rs crates/cubek-fft/src/fft/irfft_interleaved.rs \
  crates/cubek-fft/tests/fft/mod.rs crates/cubek-fft/tests/fft/interleaved_irfft.rs
git commit -m "feat(fft): add small interleaved IRFFT"
```

---

### Task 6: Large interleaved RFFT and IRFFT

**Files:**
- Modify: `crates/cubek-fft/src/fft/rfft_large.rs`
- Modify: `crates/cubek-fft/src/fft/rfft_interleaved.rs`
- Modify: `crates/cubek-fft/src/fft/irfft_interleaved.rs`
- Modify: `crates/cubek-fft/tests/fft/interleaved_rfft.rs`
- Modify: `crates/cubek-fft/tests/fft/interleaved_irfft.rs`

**Interfaces:**
- Extends Tasks 4 and 5 without public signature changes.
- Keeps packed CFFT buffers split and changes only the external post/pre boundary kernels.

- [ ] **Step 1: Add failing heavy large/padded tests**

Add the first-large-size, batched large, strided-axis large, and virtual padding cases for both directions. Derive the first large size as `2 * max_shared_fft_n(client)` rather than assuming 8192.

```rust
#[test]
#[cfg(feature = "heavy")]
fn interleaved_rfft_and_irfft_first_large_round_trip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let n_fft = first_four_step_n(&client);
    run_real_round_trip(&client, vec![2, n_fft], 1, FftNormalization::None,
        FftNormalization::ByN, 0.04);
}
```

- [ ] **Step 2: Run heavy focused tests and verify RED**

```bash
cargo test -p cubek-fft --features heavy --test lib interleaved_rfft_and_irfft_first_large_round_trip -- --nocapture
```

Expected: launch fails because Tasks 4/5 have no large interleaved dispatch.

- [ ] **Step 3: Add direct interleaved large boundary kernels**

For large RFFT, retain real-to-split packing and split CFFT. Change only the final postprocess kernel to write real/imaginary component views in one complex output, applying the selected scale.

For large IRFFT, change only the pre-process kernel to read component views into split packed input. Retain split inverse CFFT and real unpack. The existing unpack implements `ByN`; multiply its store by this adjustment:

```rust
let adjustment = match normalization {
    FftNormalization::None => n_fft as f32,
    FftNormalization::ByN => 1.0,
    FftNormalization::Ortho => (n_fft as f32).sqrt(),
};
```

This preserves the existing algebra while producing the public normalization contract. Fuse the adjustment into the unpack store. Do not add conversion or scaling launches.

- [ ] **Step 4: Run all heavy FFT tests**

```bash
cargo test -p cubek-fft --features heavy --test lib interleaved -- --nocapture
cargo test -p cubek-fft --features heavy
cargo fmt --all -- --check
```

Expected: small, four-step, padding, normalization, and existing heavy split tests all pass.

- [ ] **Step 5: Commit large RFFT/IRFFT**

```bash
git add crates/cubek-fft/src/fft/rfft_large.rs \
  crates/cubek-fft/src/fft/rfft_interleaved.rs \
  crates/cubek-fft/src/fft/irfft_interleaved.rs \
  crates/cubek-fft/tests/fft/interleaved_rfft.rs \
  crates/cubek-fft/tests/fft/interleaved_irfft.rs
git commit -m "feat(fft): add large interleaved real FFTs"
```

---

### Task 7: Benchmarks, documentation, review, PR, and merge

**Files:**
- Modify: `crates/cubek-fft/src/eval/benchmarks/strategy.rs`
- Modify: `crates/cubek-fft/src/eval/benchmarks/benchmark.rs`
- Modify: `crates/cubek-fft/src/eval/benchmarks/problem.rs`
- Modify: `crates/cubek-fft/src/lib.rs`: add crate-level ABI documentation.
- Modify: `crates/cubek-fft/tests/fft/interleaved_validation.rs`

**Interfaces:**
- Consumes the complete interleaved API.
- Produces benchmark catalogue entries and final user-facing ABI documentation.

- [ ] **Step 1: Add failing benchmark catalogue assertions**

Extend the benchmark strategy to distinguish split and interleaved:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftStrategy { Split, Interleaved }

pub fn strategies() -> Vec<CatalogEntry<FftStrategy>> {
    vec![
        CatalogEntry::new("default", "Default (split)", FftStrategy::Split),
        CatalogEntry::new("interleaved", "Interleaved C32", FftStrategy::Interleaved),
    ]
}
```

Update the existing benchmark catalogue test to assert the backward-compatible `default` ID, the new `interleaved` ID, and representative 4096/8192 problem IDs.

- [ ] **Step 2: Run benchmark catalogue tests and verify RED**

```bash
cargo test -p cubek-fft --features benchmarks --test lib bench_catalog -- --nocapture
```

Expected: the catalogue assertion fails because only the default strategy exists.

- [ ] **Step 3: Implement benchmark dispatch and ABI documentation**

Make `FftBench::prepare` allocate either split tensors or one `ComplexTensorHandle` according to `FftStrategy`. Dispatch `execute` to the corresponding split or interleaved launch API. Preserve the same seeded data and shapes so results are comparable. Include the strategy ID in the benchmark name.

Document the exact physical order, logical shape/stride units, normalization table, output uniqueness requirement, small/four-step device selector, F32/C32 backend scope, and F64/C64 deferral. State explicitly that profiling should show the algorithm kernels only and no standalone pack/unpack pass.

- [ ] **Step 4: Run fresh final verification**

Run all commands from the worktree and inspect every exit code:

```bash
cargo fmt --all -- --check
cargo clippy -p cubek-fft --all-targets --features heavy,benchmarks -- -D warnings
cargo test -p cubek-fft
cargo test -p cubek-fft --features heavy
cargo test -p cubek-fft --features benchmarks --test lib bench_catalog -- --nocapture
cargo test -p cubek-fft --features cubecl/wgpu
git diff --check origin/main...HEAD
```

On the Apple host, record the WGPU adapter as Metal in the test output or a short PR note. Expected: formatting/clippy are clean, every test command has zero failures, and the diff check emits no errors.

- [ ] **Step 5: Commit docs/benchmarks and request code review**

```bash
git add crates/cubek-fft/src/eval/benchmarks crates/cubek-fft/src/lib.rs \
  crates/cubek-fft/tests/fft/interleaved_validation.rs
git commit -m "docs(fft): document and benchmark interleaved C32"
```

Run the requesting-code-review workflow against `origin/main..HEAD`; fix every critical and important finding with a new failing regression test and a separate commit.

- [ ] **Step 6: Push and create the PR**

```bash
git push -u origin codex/fft-interleaved-c32
```

Create a PR targeting `main` titled `FFT: add interleaved F32/C32 APIs`. In the body include:

- `Refs #6` rather than `Closes #6`;
- the hybrid global-interleaved/internal-split architecture;
- normalization and typed-error behavior;
- exact local verification commands/results;
- Metal/WGPU test evidence;
- the remaining F64/C64 follow-up.

- [ ] **Step 7: Drive CI and review to merge**

Use the GitHub CI workflow to inspect every required check. Diagnose failures from logs, reproduce locally where possible, add a failing regression test, fix, rerun the full relevant command, and push normally. Address actionable review threads and keep the worktree alive.

When all required checks are green, the PR is mergeable, and no critical/important review finding remains, merge the PR using the repository's allowed merge method. Re-fetch `main`, verify the PR state is `MERGED`, and update issue #6 with the merged PR link plus an unchecked F64/C64 follow-up item. Do not close issue #6.

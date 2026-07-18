# Interleaved C32 FFT Design

## Scope

This design covers the first implementation pull request for tensor4all/cubek#6. It adds an interleaved F32/C32 global-memory ABI to CFFT, RFFT, and IRFFT on `main`. F64/C64 execution is deliberately deferred to a second pull request because Metal does not provide native `f64` shader arithmetic.

The first pull request must be complete for F32/C32: public APIs, caller-owned launches, small and four-step kernels, normalization, validation, tests, documentation, and benchmarks. The existing split real/imaginary APIs and their behavior remain unchanged.

## Chosen approach

Use hybrid direct kernels:

- Global complex input and output buffers are interleaved as `[re0, im0, re1, im1, ...]`.
- Kernel loads expose separate real and imaginary scalar views of the interleaved buffer.
- Existing split shared-memory arrays and split four-step scratch buffers remain unchanged.
- Final kernel stores write interleaved output directly.
- No standalone pack or unpack kernel is introduced.

This avoids extra global-memory passes while limiting changes to the global load/store boundaries. Fully interleaving shared memory and scratch is outside this pull request.

## Public data model

Add `ComplexTensorHandle<R>` and `ComplexTensorBinding<R>` as dedicated wrappers for complex tensors. A wrapper owns or borrows one scalar buffer while keeping these concepts distinct:

- logical complex shape;
- logical complex strides;
- scalar storage dtype;
- physical scalar buffer length.

For C32, logical complex element `i` occupies scalar offsets `2 * i` and `2 * i + 1`. Logical strides are measured in complex elements and are converted to scalar offsets at the kernel boundary. The public logical shape never contains a synthetic trailing dimension of length two.

Constructors validate rank, shape/stride rank agreement, offset and extent arithmetic, scalar alignment, and that the physical buffer covers the final reachable real/imaginary scalar. Contiguous allocation uses exactly `2 * logical_element_count * sizeof(f32)` bytes. Binding construction is fallible and never silently changes dtype or shape.

The representation must be byte-compatible with contiguous Rust `num_complex::Complex32` slices. The crate does not add a general CubeCL complex scalar type.

## Public FFT APIs

Keep all existing split APIs source-compatible. Add clearly separate interleaved APIs:

- allocating convenience functions: `cfft_interleaved`, `rfft_interleaved`, and `irfft_interleaved`;
- caller-owned functions: `cfft_interleaved_launch`, `rfft_interleaved_launch`, and `irfft_interleaved_launch`;
- padded caller-owned functions: `rfft_interleaved_launch_padded` and `irfft_interleaved_launch_padded`.

The new functions return `Result` and do not call `assert!` or `unwrap` for user input validation. The launch APIs accept caller-owned input and output bindings. They perform no host transfer and no hidden output allocation. The large path may allocate the same internal split scratch currently allocated by the split implementation; exposing reusable scratch is a later optimization, not part of this ABI change.

Out-of-place operation is guaranteed. Unsupported overlapping input/output bindings return a typed error before launch. In-place CFFT is deferred until its aliasing contract is designed independently.

The first pull request accepts only the F32/C32 dtype pair. The wrappers and enum-based API must not preclude adding F64/C64 dispatch later, but no dummy F64 path is added.

## Normalization

Add the public enum:

```rust
pub enum FftNormalization {
    None,
    ByN,
    Ortho,
}
```

The scale for an FFT of length `n_fft` is:

- `None`: `1.0`;
- `ByN`: `1.0 / n_fft`;
- `Ortho`: `1.0 / sqrt(n_fft)`.

The enum is accepted by all new forward and inverse interleaved APIs and has the same meaning in either direction. Scaling is fused into the final output store. No separate scaling kernel is launched.

Existing split APIs retain their current behavior: CFFT and RFFT are unscaled, while IRFFT scales by `1 / n_fft`.

## Kernel data flow

### Small CFFT

The interleaved input is viewed through component-aware layouts. The real layout maps a logical complex coordinate to the even scalar position and the imaginary layout maps it to the adjacent odd scalar position. Loads place values in the existing bit-reversed split shared-memory arrays. The existing butterfly implementation is unchanged. Final real and imaginary values are scaled and stored through the two component views of the interleaved output.

### Four-step CFFT

The first radix stage reads interleaved global input through component-aware views and writes the existing split real/imaginary scratch buffers. The second radix stage continues to operate in place on split scratch. The final transpose/reorder reads split scratch and writes scaled interleaved global output. There is no extra conversion pass.

### RFFT

The real F32 input and internal split shared memory are unchanged. The final half-spectrum store writes scaled values directly to the real and imaginary component views of one C32 output binding.

### IRFFT

The half-spectrum load reads real and imaginary component views from one C32 binding, reconstructs conjugate bins in split shared memory, and runs the existing inverse butterfly. The final real F32 store applies the selected normalization.

### Large RFFT and IRFFT

The current large paths lower through CFFT. Their boundary stages are changed to consume or produce the interleaved binding while retaining split internal scratch. They must not route through a standalone interleaved-to-split conversion kernel.

## Small versus large selection

Interleaved and split transforms share the existing `max_shared_fft_n(client)` selector. Do not introduce a second threshold or a hard-coded 4096 constant.

The selector derives the largest power-of-two transform whose two F32 shared arrays fit in `client.properties().hardware.max_shared_memory_size`. On a device with 32 KiB available, this yields 4096. A transform at or below the result uses the small shared-memory path; the next power of two uses the four-step path.

Four-step factorization continues to keep both factors at or below the same device-derived shared-memory limit and chooses a balanced power-of-two split.

## Validation and errors

Add a public `FftError` type with variants that distinguish at least:

- unsupported scalar dtype or dtype pair;
- rank, shape, or stride mismatch;
- transform axis out of bounds;
- non-power-of-two length or `n_fft < 2`;
- invalid `signal_len` or `spec_bins`;
- insufficient physical buffer bytes or invalid scalar alignment;
- unsupported input/output overlap;
- size arithmetic overflow;
- underlying CubeCL launch/setup failure where the lower layer returns an error.

All validation happens before the first kernel launch or scratch allocation. Empty batches remain successful no-ops after metadata validation.

## Compatibility

- Existing split symbols remain available with unchanged signatures.
- Existing split numerical and normalization behavior remains unchanged.
- Existing arbitrary transform axis, batches, strided tensor layout, and virtual padding behavior is preserved.
- The interleaved API is additive and can be adopted independently by tenferro.
- Issue #6 remains open after this pull request because F64/C64 and the backend capability matrix are handled by the second pull request.

## Testing

Follow test-driven development. Each public behavior is introduced by a failing test before production code.

Host-side tests cover:

- contiguous Complex32 byte ordering and logical shape;
- physical byte extent calculation for strided logical layouts;
- all typed validation failures;
- normalization scale calculation;
- small/four-step selector boundary using mocked hardware properties or a pure selector helper.

Backend numerical tests cover CFFT, RFFT, and IRFFT with:

- transform axis 0, a middle axis, and the last axis;
- single and batched inputs;
- contiguous and currently supported strided layouts;
- virtual padding and truncation parameters;
- `None`, `ByN`, and `Ortho` normalization;
- minimum FFT length;
- the largest small transform and first four-step transform supported by the test device;
- forward/inverse round trips and comparison with the CPU reference;
- direct inspection of interleaved output scalar order.

Run the F32/C32 suite on CubeCL WGPU/Metal locally and on the repository's CI-supported runtime. Tests that require more device memory may be categorized consistently with the existing heavy/extended test features.

Regression tests also prove that existing split APIs still compile and retain their current results.

## Benchmarks and documentation

Add interleaved CFFT/RFFT/IRFFT cases to the existing FFT benchmark catalogue. Compare them with the existing split implementation at representative small and four-step sizes. Record kernel count or profiling evidence showing that the interleaved path contains no standalone pack/unpack pass.

Document:

- the logical-versus-physical complex tensor contract;
- Rust `Complex32` byte compatibility;
- normalization semantics;
- supported F32/C32 backend behavior;
- the explicit deferral of F64/C64 to the follow-up pull request.

## Pull request and merge criteria

The pull request targets `tensor4all/cubek:main` and references issue #6 without closing it. Before merge:

- formatting and lint checks pass;
- the full `cubek-fft` unit/integration suite passes;
- WGPU/Metal F32/C32 tests pass on the Apple development host;
- repository-required GitHub Actions checks pass;
- no unresolved critical or important review findings remain;
- the issue checklist is updated to identify the merged F32/C32 phase and the remaining F64/C64 phase.

## Deferred work

The follow-up pull request adds F64/C64 to the same API and layout model on runtimes with native F64 support. Unsupported runtimes, including Metal, must reject F64/C64 before launch with a typed capability error. It must not narrow to F32, stage through the host, or fall back to CPU execution.

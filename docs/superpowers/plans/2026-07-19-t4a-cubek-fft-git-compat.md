# CubeK FFT Git Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the merged interleaved F32/C32 CubeK FFT implementation available as a git dependency on the tensor4all CubeCL 0.10 fork used by tenferro.

**Architecture:** Backport only the `cubek-fft` implementation and tests from CubeK main onto `release/t4a-0.2.0`, keep the package private/non-released, and adapt the small amount of newer CubeCL syntax to the 0.10 API (`SharedMemory` and 0.10 tensor views). Add the crate to the workspace so inherited metadata/dependencies resolve for git consumers.

**Tech Stack:** Rust, CubeK FFT, `t4a-cubecl` 0.10, CubeCL CPU test runtime.

## Global Constraints

- Preserve the merged public interleaved APIs: `ComplexTensorHandle`, `FftNormalization`, `cfft_interleaved*`, `rfft_interleaved*`, and `irfft_interleaved*`.
- Support only F32 real and interleaved C32 complex tensors; do not add F64/C64.
- Preserve the same device-derived threshold for small and large CFFT/RFFT/IRFFT paths.
- Do not add standalone conversion or scaling kernels.
- Keep `cubek-fft` unpublished (`publish = false`); tenferro consumes this branch by git revision.
- Use the `t4a-cubecl` 0.10 workspace dependencies and `t4a-cubek-test-utils` 0.2 package.
- Do not backport the newer benchmark-catalog framework, which is not available in `t4a-cubek-test-utils` 0.2 and is not required by tenferro runtime execution.
- Do not merge or release from this task.

---

### Task 1: Backport interleaved FFT to the t4a 0.2 dependency line

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/cubek-fft/**`
- Create: `docs/superpowers/plans/2026-07-19-t4a-cubek-fft-git-compat.md`

**Interfaces:**
- Produces a git-resolvable private `cubek-fft` package using the same `t4a-cubecl` types as tenferro.
- Produces the same public FFT API and F32/C32 semantics as CubeK PRs #7 and #8.

- [ ] **Step 1: Add `crates/cubek-fft` to the t4a workspace**

  Add the crate to `workspace.members` and remove it from `workspace.exclude`. Keep its package name `cubek-fft`, set `publish = false`, point repository metadata at `tensor4all/cubek`, and use `t4a-cubek-test-utils =0.2.0`.

- [ ] **Step 2: Backport the merged CubeK FFT source and tests**

  Bring the runtime portion of `crates/cubek-fft` from commit `e66fe2e60f9dfb4c77be5c68cc9979bbdbabc354`, including interleaved ABI validation, CFFT/RFFT/IRFFT kernels, normalization, small/large selection, CPU references, and correctness tests. Omit the newer benchmark-catalog modules that require post-0.2 test utilities.

- [ ] **Step 3: Adapt only CubeCL API syntax**

  Replace newer `Shared<[F]>`/`Shared::new_slice` usage with CubeCL 0.10 `SharedMemory<F>`/`SharedMemory::new`, and newer mutable-view aliases with `View<F, Coords1d, ReadWrite>`. Do not alter algorithms, thresholds, layouts, normalization, or launch signatures.

- [ ] **Step 4: Verify compilation and correctness**

  Run:

  ```bash
  cargo fmt --all -- --check
  cargo check -p cubek-fft
  cargo test -p cubek-fft
  cargo clippy -p cubek-fft --all-targets --all-features -- -D warnings
  ```

  Expected: all commands pass, including small and large interleaved CPU-runtime tests.

- [ ] **Step 5: Commit**

  ```bash
  git add Cargo.toml Cargo.lock crates/cubek-fft docs/superpowers/plans/2026-07-19-t4a-cubek-fft-git-compat.md
  git commit -m "feat(fft): backport interleaved APIs to t4a CubeCL"
  ```

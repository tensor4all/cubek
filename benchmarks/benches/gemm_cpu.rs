//! Focused CPU GEMM comparison: CpuGemm against the unit matmuls (its real
//! fallback competitors), over a few CPU-sized shapes.
//!
//! `cargo bench -p benchmarks --bench gemm_cpu --features cpu`
//!
//! The `cpu` feature is **required**: without a backend feature flag the bench
//! silently selects `WgpuRuntime` (Metal on macOS) even though it's named "cpu",
//! and the numbers reflect the GPU, not the CPU (masked reads cost ~170× on
//! Metal vs ~1.4× on CPU — entirely different conclusions). `--features cpu`
//! pins `TestRuntime = CpuRuntime` (the MLIR/LLVM CPU JIT).
//!
//! On the CPU runtime CpuGemm beats both unit matmuls on every shape here
//! (512³ ~2×, vecmat ~3-6×). The forced-tile probes (`cpu_gemm_t32/48/64`)
//! isolate the edge-masking cost on the aligned 512³ case — a divisor tile
//! (t32/t64) needs no masking and runs ~1.4-1.6× faster than the non-divisor
//! default. They are pointless (and slow) on the degenerate vecmat/matvec
//! shapes, where forcing a large tile onto a size-1 axis just wastes compute, so
//! they're scoped to the square problem below.
//!
//! Unit `*_max` tiles and the plane/cmma strategies are excluded: they request
//! more shared memory than the CPU exposes and *panic* during profiling (an
//! uncatchable abort across the launch boundary).

/// CpuGemm (default heuristic) vs the unit matmuls, across square + degenerate shapes.
const STRATEGIES: &[&str] = &["cpu_gemm", "simple_unit_min", "double_unit_min"];
const PROBLEMS: &[&str] = &[
    "rect_1x512x512x512_rr_f32",
    "vecmat_2x1x4096x4096_rr_f32",
    "matvec_2x8192x1x8192_rr_f32",
];

/// Masked vs maskless probe: forced divisor (t32/t64, maskless on 512) vs non-divisor
/// (t48, masked) vs the heuristic default, on the one aligned square shape.
const MASK_STRATEGIES: &[&str] = &["cpu_gemm", "cpu_gemm_t48", "cpu_gemm_t64", "cpu_gemm_t32"];
const MASK_PROBLEMS: &[&str] = &["rect_1x512x512x512_rr_f32"];

fn main() {
    let category = &benchmarks::gemm::Category;
    benchmarks::run_category_filtered(category, STRATEGIES, PROBLEMS);
    benchmarks::run_category_filtered(category, MASK_STRATEGIES, MASK_PROBLEMS);
}

//! Focused CPU GEMM comparison: CpuGemm

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::eval::benchmarks::gemm::{self, GemmProblem};
use crate::strategy::Strategy;

/// CpuGemm vs the unit matmuls, plus the forced-tile mask probe. The forced tiles
/// are diagnostic on the 512 square: `t64`/`t32` divide 512 (maskless), `t48` does
/// not (masked).
const STRATEGIES: &[&str] = &[
    "cpu_gemm",
    "simple_unit_min",
    "double_unit_min",
    "cpu_gemm_t48",
    "cpu_gemm_t64",
    "cpu_gemm_t32",
];

/// Aligned square, a vector × matrix, and a matrix × vector — CPU-sized shapes.
const PROBLEMS: &[&str] = &[
    "rect_1x512x512x512_rr_f32",
    "vecmat_2x1x4096x4096_rr_f32",
    "matvec_2x8192x1x8192_rr_f32",
];

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = GemmProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "gemm_cpu"
    }

    fn label(&self) -> &'static str {
        "GEMM (CPU)"
    }

    fn problems(&self) -> Vec<CatalogEntry<GemmProblem>> {
        gemm::problems()
            .into_iter()
            .filter(|p| PROBLEMS.contains(&p.id.as_str()))
            .collect()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        gemm::strategies()
            .into_iter()
            .filter(|s| STRATEGIES.contains(&s.id.as_str()))
            .collect()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &GemmProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        gemm::bench(strategy, problem, num_samples)
    }

    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = GemmProblem, Strategy = Strategy>>
    {
        Some(&gemm::GemmCorrectness)
    }
}

use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{HostData, Progress};

use crate::eval::benchmarks::problem::{CfftProblem, FftProblem};
use crate::eval::benchmarks::strategy::FftStrategy;
use crate::eval::cpu_reference::{
    complex_kernel_result, cpu_reference_complex_result, cpu_reference_result,
    interleaved_kernel_result, kernel_result as fft_kernel_result,
};

pub struct FftCorrectness;
pub struct CfftCorrectness;

#[derive(Debug, PartialEq, Eq)]
enum CorrectnessKernel {
    Split,
    Interleaved,
}

fn kernel_backend(strategy: FftStrategy) -> CorrectnessKernel {
    match strategy {
        FftStrategy::Split => CorrectnessKernel::Split,
        FftStrategy::Interleaved => CorrectnessKernel::Interleaved,
    }
}

impl cubek_test_utils::Correctness for FftCorrectness {
    type Problem = FftProblem;
    type Strategy = FftStrategy;

    fn kernel_result(
        &self,
        strategy: &FftStrategy,
        problem: &FftProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        match kernel_backend(*strategy) {
            CorrectnessKernel::Split => fft_kernel_result(
                client,
                problem.shape.clone(),
                dim,
                problem.mode,
                seeds[0],
                seeds[1],
            ),
            CorrectnessKernel::Interleaved => interleaved_kernel_result(
                client,
                problem.shape.clone(),
                dim,
                problem.mode,
                seeds[0],
                seeds[1],
            ),
        }
    }

    fn reference_result(
        &self,
        problem: &FftProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        cpu_reference_result(
            client,
            problem.shape.clone(),
            dim,
            problem.mode,
            seeds[0],
            seeds[1],
            progress,
        )
    }
}

impl cubek_test_utils::Correctness for CfftCorrectness {
    type Problem = CfftProblem;
    type Strategy = FftStrategy;

    fn kernel_result(
        &self,
        strategy: &FftStrategy,
        problem: &CfftProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        complex_kernel_result(
            client,
            problem.shape.clone(),
            dim,
            problem.mode,
            seeds[0],
            seeds[1],
            kernel_backend(*strategy) == CorrectnessKernel::Interleaved,
        )
    }

    fn reference_result(
        &self,
        problem: &CfftProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dim = problem.shape.len() - 1;
        Ok(cpu_reference_complex_result(
            client,
            problem.shape.clone(),
            dim,
            problem.mode,
            seeds[0],
            seeds[1],
            progress,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_strategy_selects_interleaved_correctness_kernel() {
        assert_eq!(
            kernel_backend(FftStrategy::Interleaved),
            CorrectnessKernel::Interleaved
        );
    }
}

use cubek_test_utils::CatalogEntry;

use crate::FftMode;

/// Real FFT benchmark problem. Kept source-compatible with the original
/// benchmark API so downstream users can construct it with two public fields.
pub struct FftProblem {
    pub shape: Vec<usize>,
    pub mode: FftMode,
}

/// Complex-to-complex FFT benchmark problem, exposed through [`CfftCategory`](super::CfftCategory).
pub struct CfftProblem {
    pub shape: Vec<usize>,
    pub mode: FftMode,
}

pub fn problems() -> Vec<CatalogEntry<FftProblem>> {
    vec![
        fft_problem(
            "forward_5x2x2048",
            "Forward (5x2x2048)",
            vec![5, 2, 2048],
            FftMode::Forward,
        ),
        fft_problem(
            "inverse_5x2x2048",
            "Inverse (5x2x2048)",
            vec![5, 2, 2048],
            FftMode::Inverse,
        ),
        fft_problem(
            "forward_128x2048",
            "Forward (128x2048)",
            vec![128, 2048],
            FftMode::Forward,
        ),
        fft_problem(
            "inverse_128x2048",
            "Inverse (128x2048)",
            vec![128, 2048],
            FftMode::Inverse,
        ),
        fft_problem(
            "forward_1x4096",
            "Forward (1x4096)",
            vec![1, 4096],
            FftMode::Forward,
        ),
        fft_problem(
            "inverse_1x4096",
            "Inverse (1x4096)",
            vec![1, 4096],
            FftMode::Inverse,
        ),
        fft_problem(
            "forward_1x8192",
            "Forward (1x8192)",
            vec![1, 8192],
            FftMode::Forward,
        ),
        fft_problem(
            "inverse_1x8192",
            "Inverse (1x8192)",
            vec![1, 8192],
            FftMode::Inverse,
        ),
        fft_problem(
            "forward_1x16384",
            "Forward (1x16384)",
            vec![1, 16384],
            FftMode::Forward,
        ),
        fft_problem(
            "inverse_1x16384",
            "Inverse (1x16384)",
            vec![1, 16384],
            FftMode::Inverse,
        ),
    ]
}

pub fn cfft_problems() -> Vec<CatalogEntry<CfftProblem>> {
    vec![
        cfft_problem(
            "forward_1x4096",
            "Forward (1x4096)",
            vec![1, 4096],
            FftMode::Forward,
        ),
        cfft_problem(
            "inverse_1x4096",
            "Inverse (1x4096)",
            vec![1, 4096],
            FftMode::Inverse,
        ),
        cfft_problem(
            "forward_1x8192",
            "Forward (1x8192)",
            vec![1, 8192],
            FftMode::Forward,
        ),
        cfft_problem(
            "inverse_1x8192",
            "Inverse (1x8192)",
            vec![1, 8192],
            FftMode::Inverse,
        ),
    ]
}

fn fft_problem(
    id: &str,
    label: &str,
    shape: Vec<usize>,
    mode: FftMode,
) -> CatalogEntry<FftProblem> {
    CatalogEntry::new(id, label, FftProblem { shape, mode })
}

fn cfft_problem(
    id: &str,
    label: &str,
    shape: Vec<usize>,
    mode: FftMode,
) -> CatalogEntry<CfftProblem> {
    CatalogEntry::new(id, label, CfftProblem { shape, mode })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_problem_remains_constructible_with_shape_and_mode_only() {
        let problem = FftProblem {
            shape: vec![1, 8],
            mode: FftMode::Forward,
        };
        assert_eq!(problem.shape, [1, 8]);
        assert_eq!(problem.mode, FftMode::Forward);
        assert!(
            problems()
                .iter()
                .all(|entry| !entry.id.starts_with("cfft_"))
        );
    }

    #[test]
    fn cfft_catalog_keeps_small_and_four_step_cases_separate() {
        let ids = cfft_problems()
            .into_iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"forward_1x4096".to_string()));
        assert!(ids.contains(&"inverse_1x8192".to_string()));
    }
}

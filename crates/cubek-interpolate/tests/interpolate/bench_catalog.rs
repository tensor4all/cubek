//! Correctness over the interpolate benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_interpolate::definition::InterpolateProblem;
use cubek_interpolate::eval::benchmarks::{InterpolateCorrectness, InterpolateStrategy};
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

const INTERP_EPS: f32 = 1e-3;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    use cubek_interpolate::eval::benchmarks::{problems, strategies};

    let strategy: InterpolateStrategy = lookup(strategies(), strategy_id);
    let problem: InterpolateProblem = lookup(problems(), problem_id);

    let actual = match InterpolateCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = InterpolateCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, INTERP_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
#[ignore = "TODO - FAILS"]
fn nearest_upsample_2x_64x64_default() {
    run("default", "NEAREST_UPSAMPLE_2X_64X64");
}

#[test]
#[ignore = "TODO - FAILS"]
fn nearest_downsample_2x_256x256_default() {
    run("default", "NEAREST_DOWNSAMPLE_2X_256X256");
}

#[test]
#[ignore = "TODO - FAILS"]
fn nearest_upsample_4x_512x512_default() {
    run("default", "NEAREST_UPSAMPLE_4X_512X512");
}

#[test]
#[ignore = "TODO - FAILS"]
fn nearest_downsample_4x_2048x2048_default() {
    run("default", "NEAREST_DOWNSAMPLE_4X_2048X2048");
}

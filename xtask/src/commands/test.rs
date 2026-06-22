use std::collections::HashMap;

use tracel_xtask::prelude::*;

const CI_WGPU_NO_RUN_ENV: &str = "CUBEK_CI_WGPU_NO_RUN";

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeKTestCmdArgs {
    /// Kept for CI workflow compatibility; tests already target the publish closure.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    args: CubeKTestCmdArgs,
    _env: Environment,
    _context: Context,
) -> anyhow::Result<()> {
    let backends: &[&str] = &["cubecl/wgpu"];
    let envs = args.ci.then(|| HashMap::from([("RUST_TEST_THREADS", "1")]));
    let no_run = std::env::var_os(CI_WGPU_NO_RUN_ENV).is_some();
    for backend in backends {
        let mut test_args = vec!["--features", *backend];
        if no_run {
            test_args.push("--no-run");
        }
        let group_msg = if no_run {
            format!("Compile tests on backend {backend:?}")
        } else {
            format!("Test on backend {backend:?}")
        };
        helpers::custom_crates_tests(
            vec![
                "t4a-cubek-matmul",
                "t4a-cubek-quant",
                "t4a-cubek-random",
                "t4a-cubek-std",
                "t4a-cubek-test-utils",
            ],
            test_args,
            envs.clone(),
            None,
            &group_msg,
        )?;
    }
    Ok(())
}

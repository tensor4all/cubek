use tracel_xtask::prelude::*;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeKTestCmdArgs {
    /// Kept for CI workflow compatibility; tests already target the publish closure.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    _args: CubeKTestCmdArgs,
    _env: Environment,
    _context: Context,
) -> anyhow::Result<()> {
    let backends: &[&str] = &["cubecl/wgpu"];
    for backend in backends {
        helpers::custom_crates_tests(
            vec![
                "t4a-cubek-matmul",
                "t4a-cubek-quant",
                "t4a-cubek-random",
                "t4a-cubek-std",
                "t4a-cubek-test-utils",
            ],
            vec!["--features", backend],
            None,
            None,
            &format!("Test on backend {backend:?}"),
        )?;
    }
    Ok(())
}

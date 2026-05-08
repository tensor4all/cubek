use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::eval::benchmarks::strategy::InterpolateStrategy;
use crate::definition::InterpolateProblem;
use crate::interpolate;

pub fn bench(
    _strategy: &InterpolateStrategy,
    problem: &InterpolateProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let bench = InterpolateBench {
        problem: problem.clone(),
        device,
        client,
        dtype,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct InterpolateBench {
    problem: InterpolateProblem,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtype: StorageType,
    samples: usize,
}

impl Benchmark for InterpolateBench {
    type Input = TensorHandle<TestRuntime>;
    type Output = TensorHandle<TestRuntime>;

    fn prepare(&self) -> Self::Input {
        TestInput::builder(self.client.clone(), Shape::new(self.problem.input_shape))
            .dtype(self.dtype)
            .uniform(0, -1., 1.)
            .generate_without_host_data()
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<TestRuntime>, String> {
        let [n, _, _, c] = self.problem.input_shape;
        let output_shape = vec![
            n,
            self.problem.output_size[0],
            self.problem.output_size[1],
            c,
        ];
        let output = TensorHandle::empty(&self.client, output_shape, self.dtype);

        interpolate(
            &self.client,
            input.binding(),
            output.clone().binding(),
            self.problem.options.clone(),
            self.dtype,
        )
        .map_err(|err| format!("{err}"))?;

        Ok(output)
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "interpolate-{:?}-{:?}-{:?}-{:?}",
            self.dtype, self.problem.options.mode, self.device, self.problem.input_shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

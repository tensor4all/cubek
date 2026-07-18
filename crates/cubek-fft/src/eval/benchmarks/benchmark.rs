use std::marker::PhantomData;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, StridedLayout, TestInput};

use crate::eval::benchmarks::problem::FftProblem;
use crate::eval::benchmarks::strategy::FftStrategy;
use crate::{
    ComplexTensorHandle, FftMode, FftNormalization, irfft_interleaved_launch, irfft_launch,
    rfft_interleaved_launch, rfft_launch,
};

pub fn bench(
    strategy: &FftStrategy,
    problem: &FftProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = FftBench::<f32> {
        shape: problem.shape.clone(),
        mode: problem.mode,
        strategy: *strategy,
        device,
        client,
        samples: num_samples,
        _e: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct FftBench<E> {
    shape: Vec<usize>,
    mode: FftMode,
    strategy: FftStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

#[derive(Clone)]
enum FftInput {
    Split {
        signal: TensorHandle<TestRuntime>,
        spectrum_re: TensorHandle<TestRuntime>,
        spectrum_im: TensorHandle<TestRuntime>,
    },
    Interleaved {
        signal: TensorHandle<TestRuntime>,
        spectrum: ComplexTensorHandle<TestRuntime>,
    },
}

fn make_uniform(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::from(shape))
        .layout(StridedLayout::RowMajor)
        .dtype(dtype)
        .uniform(seed, 0., 1.)
        .generate_without_host_data()
}

fn empty_handle(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    elem: cubecl::ir::Type,
) -> TensorHandle<TestRuntime> {
    TensorHandle::empty(client, shape, elem)
}

fn make_interleaved_uniform(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
    re_seed: u64,
    im_seed: u64,
) -> ComplexTensorHandle<TestRuntime> {
    let real = TestInput::builder(client.clone(), Shape::from(shape.clone()))
        .layout(StridedLayout::RowMajor)
        .dtype(dtype)
        .uniform(re_seed, 0., 1.)
        .f32_host_data();
    let imaginary = TestInput::builder(client.clone(), Shape::from(shape.clone()))
        .layout(StridedLayout::RowMajor)
        .dtype(dtype)
        .uniform(im_seed, 0., 1.)
        .f32_host_data();
    let values = real
        .iter_indexed_f32()
        .zip(imaginary.iter_indexed_f32())
        .flat_map(|((_, re), (_, im))| [re, im])
        .collect();

    let mut physical_shape = shape.clone();
    let last = physical_shape.len() - 1;
    physical_shape[last] *= 2;
    let physical = TestInput::builder(client.clone(), Shape::from(physical_shape))
        .layout(StridedLayout::RowMajor)
        .dtype(dtype)
        .custom(values)
        .generate_without_host_data();
    ComplexTensorHandle::new_contiguous(shape, physical.handle, dtype)
        .expect("benchmark input must use a supported C32 layout")
}

impl<E: Float> Benchmark for FftBench<E> {
    type Input = FftInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let elem = E::as_type_native_unchecked();
        let storage = elem.storage_type();

        let mut shape_out = self.shape.clone();
        let dim = self.shape.len() - 1;
        shape_out[dim] = self.shape[dim] / 2 + 1;

        match self.strategy {
            FftStrategy::Split => match self.mode {
                FftMode::Forward => FftInput::Split {
                    signal: make_uniform(&client, self.shape.clone(), storage, 0),
                    spectrum_re: empty_handle(&client, shape_out.clone(), elem),
                    spectrum_im: empty_handle(&client, shape_out, elem),
                },
                FftMode::Inverse => FftInput::Split {
                    signal: empty_handle(&client, self.shape.clone(), elem),
                    spectrum_re: make_uniform(&client, shape_out.clone(), storage, 0),
                    spectrum_im: make_uniform(&client, shape_out, storage, 1),
                },
            },
            FftStrategy::Interleaved => match self.mode {
                FftMode::Forward => FftInput::Interleaved {
                    signal: make_uniform(&client, self.shape.clone(), storage, 0),
                    spectrum: ComplexTensorHandle::empty(&client, shape_out, storage)
                        .expect("benchmark output must use a supported C32 layout"),
                },
                FftMode::Inverse => FftInput::Interleaved {
                    signal: empty_handle(&client, self.shape.clone(), elem),
                    spectrum: make_interleaved_uniform(&client, shape_out, storage, 0, 1),
                },
            },
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim = self.shape.len() - 1;
        match input {
            FftInput::Split {
                signal,
                spectrum_re,
                spectrum_im,
            } => match self.mode {
                FftMode::Forward => rfft_launch(
                    &self.client,
                    signal.binding(),
                    spectrum_re.binding(),
                    spectrum_im.binding(),
                    dim,
                    E::as_type_native_unchecked().storage_type(),
                )
                .map_err(|err| format!("{err}"))?,
                FftMode::Inverse => irfft_launch(
                    &self.client,
                    spectrum_re.binding(),
                    spectrum_im.binding(),
                    signal.binding(),
                    dim,
                    E::as_type_native_unchecked().storage_type(),
                )
                .map_err(|err| format!("{err}"))?,
            },
            FftInput::Interleaved { signal, spectrum } => match self.mode {
                FftMode::Forward => rfft_interleaved_launch(
                    &self.client,
                    &signal,
                    spectrum.binding(),
                    dim,
                    FftNormalization::None,
                )
                .map_err(|err| format!("{err}"))?,
                FftMode::Inverse => irfft_interleaved_launch(
                    &self.client,
                    spectrum.binding(),
                    &signal,
                    dim,
                    FftNormalization::None,
                )
                .map_err(|err| format!("{err}"))?,
            },
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "fft-{}-{}-{:?}-{:?}",
            E::as_type_native_unchecked(),
            self.strategy.id(),
            self.shape,
            self.mode,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

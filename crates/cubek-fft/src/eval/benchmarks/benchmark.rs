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

use crate::eval::benchmarks::problem::{FftProblem, FftTransform};
use crate::eval::benchmarks::strategy::FftStrategy;
use crate::{
    ComplexTensorHandle, FftMode, FftNormalization, cfft_interleaved_launch,
    fft::cfft::{CfftBindings, cfft_launch_any_size},
    irfft_interleaved_launch, irfft_launch, rfft_interleaved_launch, rfft_launch,
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
        transform: problem.transform,
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
    transform: FftTransform,
    strategy: FftStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

enum FftInput {
    SplitForward {
        client: ComputeClient<TestRuntime>,
        signal: TensorHandle<TestRuntime>,
        spectrum_re: TensorHandle<TestRuntime>,
        spectrum_im: TensorHandle<TestRuntime>,
        elem: cubecl::ir::Type,
    },
    SplitInverse {
        client: ComputeClient<TestRuntime>,
        signal: TensorHandle<TestRuntime>,
        spectrum_re: TensorHandle<TestRuntime>,
        spectrum_im: TensorHandle<TestRuntime>,
        elem: cubecl::ir::Type,
    },
    InterleavedForward {
        client: ComputeClient<TestRuntime>,
        signal: TensorHandle<TestRuntime>,
        spectrum: ComplexTensorHandle<TestRuntime>,
    },
    InterleavedInverse {
        client: ComputeClient<TestRuntime>,
        signal: TensorHandle<TestRuntime>,
        spectrum: ComplexTensorHandle<TestRuntime>,
        elem: cubecl::ir::Type,
    },
    SplitComplex {
        client: ComputeClient<TestRuntime>,
        input_re: TensorHandle<TestRuntime>,
        input_im: TensorHandle<TestRuntime>,
        output_re: TensorHandle<TestRuntime>,
        output_im: TensorHandle<TestRuntime>,
        elem: cubecl::ir::Type,
    },
    InterleavedComplex {
        client: ComputeClient<TestRuntime>,
        input: ComplexTensorHandle<TestRuntime>,
        output: ComplexTensorHandle<TestRuntime>,
    },
}

impl Clone for FftInput {
    fn clone(&self) -> Self {
        match self {
            Self::SplitForward {
                client,
                signal,
                spectrum_re,
                spectrum_im,
                elem,
            } => Self::SplitForward {
                client: client.clone(),
                signal: signal.clone(),
                spectrum_re: empty_handle(client, spectrum_re.shape().to_vec(), *elem),
                spectrum_im: empty_handle(client, spectrum_im.shape().to_vec(), *elem),
                elem: *elem,
            },
            Self::SplitInverse {
                client,
                signal,
                spectrum_re,
                spectrum_im,
                elem,
            } => Self::SplitInverse {
                client: client.clone(),
                signal: empty_handle(client, signal.shape().to_vec(), *elem),
                spectrum_re: spectrum_re.clone(),
                spectrum_im: spectrum_im.clone(),
                elem: *elem,
            },
            Self::InterleavedForward {
                client,
                signal,
                spectrum,
            } => Self::InterleavedForward {
                client: client.clone(),
                signal: signal.clone(),
                spectrum: ComplexTensorHandle::empty(
                    client,
                    spectrum.shape().to_vec(),
                    spectrum.dtype(),
                )
                .expect("benchmark output must use a supported C32 layout"),
            },
            Self::InterleavedInverse {
                client,
                signal,
                spectrum,
                elem,
            } => Self::InterleavedInverse {
                client: client.clone(),
                signal: empty_handle(client, signal.shape().to_vec(), *elem),
                spectrum: spectrum.clone(),
                elem: *elem,
            },
            Self::SplitComplex {
                client,
                input_re,
                input_im,
                output_re,
                output_im,
                elem,
            } => Self::SplitComplex {
                client: client.clone(),
                input_re: input_re.clone(),
                input_im: input_im.clone(),
                output_re: empty_handle(client, output_re.shape().to_vec(), *elem),
                output_im: empty_handle(client, output_im.shape().to_vec(), *elem),
                elem: *elem,
            },
            Self::InterleavedComplex {
                client,
                input,
                output,
            } => Self::InterleavedComplex {
                client: client.clone(),
                input: input.clone(),
                output: ComplexTensorHandle::empty(client, output.shape().to_vec(), output.dtype())
                    .expect("benchmark output must use a supported C32 layout"),
            },
        }
    }
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

        let dim = self.shape.len() - 1;

        if self.transform == FftTransform::Complex {
            return match self.strategy {
                FftStrategy::Split => FftInput::SplitComplex {
                    client: client.clone(),
                    input_re: make_uniform(&client, self.shape.clone(), storage, 0),
                    input_im: make_uniform(&client, self.shape.clone(), storage, 1),
                    output_re: empty_handle(&client, self.shape.clone(), elem),
                    output_im: empty_handle(&client, self.shape.clone(), elem),
                    elem,
                },
                FftStrategy::Interleaved => FftInput::InterleavedComplex {
                    client: client.clone(),
                    input: make_interleaved_uniform(&client, self.shape.clone(), storage, 0, 1),
                    output: ComplexTensorHandle::empty(&client, self.shape.clone(), storage)
                        .expect("benchmark output must use a supported C32 layout"),
                },
            };
        }

        let mut shape_out = self.shape.clone();
        shape_out[dim] = self.shape[dim] / 2 + 1;

        match self.strategy {
            FftStrategy::Split => match self.mode {
                FftMode::Forward => FftInput::SplitForward {
                    client: client.clone(),
                    signal: make_uniform(&client, self.shape.clone(), storage, 0),
                    spectrum_re: empty_handle(&client, shape_out.clone(), elem),
                    spectrum_im: empty_handle(&client, shape_out, elem),
                    elem,
                },
                FftMode::Inverse => FftInput::SplitInverse {
                    client: client.clone(),
                    signal: empty_handle(&client, self.shape.clone(), elem),
                    spectrum_re: make_uniform(&client, shape_out.clone(), storage, 0),
                    spectrum_im: make_uniform(&client, shape_out, storage, 1),
                    elem,
                },
            },
            FftStrategy::Interleaved => match self.mode {
                FftMode::Forward => FftInput::InterleavedForward {
                    client: client.clone(),
                    signal: make_uniform(&client, self.shape.clone(), storage, 0),
                    spectrum: ComplexTensorHandle::empty(&client, shape_out, storage)
                        .expect("benchmark output must use a supported C32 layout"),
                },
                FftMode::Inverse => FftInput::InterleavedInverse {
                    client: client.clone(),
                    signal: empty_handle(&client, self.shape.clone(), elem),
                    spectrum: make_interleaved_uniform(&client, shape_out, storage, 0, 1),
                    elem,
                },
            },
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim = self.shape.len() - 1;
        match input {
            FftInput::SplitForward {
                signal,
                spectrum_re,
                spectrum_im,
                ..
            } => rfft_launch(
                &self.client,
                signal.binding(),
                spectrum_re.binding(),
                spectrum_im.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
            FftInput::SplitInverse {
                signal,
                spectrum_re,
                spectrum_im,
                ..
            } => irfft_launch(
                &self.client,
                spectrum_re.binding(),
                spectrum_im.binding(),
                signal.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
            FftInput::InterleavedForward {
                signal, spectrum, ..
            } => rfft_interleaved_launch(
                &self.client,
                &signal,
                spectrum.binding(),
                dim,
                FftNormalization::None,
            )
            .map_err(|err| format!("{err}"))?,
            FftInput::InterleavedInverse {
                signal, spectrum, ..
            } => irfft_interleaved_launch(
                &self.client,
                spectrum.binding(),
                &signal,
                dim,
                FftNormalization::ByN,
            )
            .map_err(|err| format!("{err}"))?,
            FftInput::SplitComplex {
                input_re,
                input_im,
                output_re,
                output_im,
                ..
            } => cfft_launch_any_size(
                &self.client,
                CfftBindings {
                    input_re: input_re.binding(),
                    input_im: input_im.binding(),
                    output_re: output_re.binding(),
                    output_im: output_im.binding(),
                },
                dim,
                E::as_type_native_unchecked().storage_type(),
                self.mode,
            )
            .map_err(|err| format!("{err}"))?,
            FftInput::InterleavedComplex { input, output, .. } => cfft_interleaved_launch(
                &self.client,
                input.binding(),
                output.binding(),
                dim,
                self.mode,
                FftNormalization::None,
            )
            .map_err(|err| format!("{err}"))?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn interleaved_bench(mode: FftMode) -> FftBench<f32> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        FftBench {
            shape: vec![1, 8],
            mode,
            transform: FftTransform::Real,
            strategy: FftStrategy::Interleaved,
            device,
            client,
            samples: 1,
            _e: PhantomData,
        }
    }

    #[test]
    fn cloned_interleaved_forward_sample_has_fresh_writable_output() {
        let bench = interleaved_bench(FftMode::Forward);
        let prepared = bench.prepare();
        bench.execute(prepared.clone()).unwrap();
    }

    #[test]
    fn cloned_interleaved_inverse_sample_has_fresh_writable_output() {
        let bench = interleaved_bench(FftMode::Inverse);
        let prepared = bench.prepare();
        bench.execute(prepared.clone()).unwrap();
    }

    #[test]
    fn cloned_interleaved_cfft_sample_has_fresh_writable_output() {
        let mut bench = interleaved_bench(FftMode::Forward);
        bench.transform = FftTransform::Complex;
        let prepared = bench.prepare();
        bench.execute(prepared.clone()).unwrap();
    }
}

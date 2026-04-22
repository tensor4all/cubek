use crate::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        instructions::{Accumulator, AccumulatorFormat, Value, ValueExpand},
        layout::ReduceOutputLayout,
    },
};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d, r#virtual::VirtualTensor},
};

#[derive(CubeType)]
/// Abstract how data is written to global memory.
///
/// Depending on the problem kind, writes might be buffered to optimize vectorization, only
/// happening when [Writer::commit()] is called.
pub enum Writer<Out: NumericVector> {
    Parallel(ParallelWriter<Out>),
    Perpendicular(PerpendicularWriter<Out>),
}

/// Build a [`ReduceOutputLayout`] from the output tensor and reduce/vec axes.
///
/// For simple reduces (`accumulator_length == 1`) `k_iter` never advances, so
/// the layout degenerates to `position = write_index` — a flat enumeration of
/// output vectors.
///
/// For topk-style reduces (`accumulator_length > 1`) the k slots live along
/// `reduce_axis` with stride `stride(reduce_axis) / vec` vectors, and the vec
/// axis (contiguous in the output, scalar stride 1) advances by one vector
/// per step, so `write_stride = 1`. When the two axes coincide — i.e. a
/// rank-1 output or any degenerate case where there is no separate SIMD axis
/// — `write_stride` collapses to `0` and `num_writes` to `1`, so every
/// `write_index` lands on the same k slot (should not matter because this collapse happens
/// only if we have only one unit).
#[cube]
fn build_reduce_output_layout<Out: NumericVector>(
    output: &VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] accumulator_length: usize,
) -> ReduceOutputLayout {
    let vec = output.vector_size();
    let num_vectored_reductions = output.shape(out_vec_axis) / vec;

    if comptime![accumulator_length == 1] {
        // Simple reduce: `write_index` is a flat vector offset into the
        // output, which matches the pre-topk behavior.
        ReduceOutputLayout::new(
            num_vectored_reductions,
            1,
            num_vectored_reductions,
            accumulator_length,
        )
    } else {
        let k_stride = output.stride(reduce_axis) / vec;
        // Branchless: `distinct` is 1 when reduce_axis != out_vec_axis, else 0.
        let distinct = usize::cast_from(reduce_axis != out_vec_axis);
        let write_stride = distinct;
        let num_writes = distinct * num_vectored_reductions + (1 - distinct);
        ReduceOutputLayout::new(k_stride, write_stride, num_writes, accumulator_length)
    }
}

#[cube]
impl<Out: NumericVector> Writer<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] acc_format: AccumulatorFormat,
    ) -> Writer<Out> {
        match vectorization_mode {
            VectorizationMode::Parallel => {
                Writer::<Out>::new_Parallel(ParallelWriter::<Out>::new::<P>(
                    input,
                    output,
                    reduce_axis,
                    out_vec_axis,
                    write_index,
                    acc_format,
                ))
            }
            VectorizationMode::Perpendicular => {
                Writer::<Out>::new_Perpendicular(PerpendicularWriter::<Out>::new::<P>(
                    input,
                    output,
                    reduce_axis,
                    out_vec_axis,
                    write_index,
                    acc_format,
                ))
            }
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        match self {
            Writer::Parallel(writer) => writer.write::<P, I>(local_index, accumulator, inst),
            Writer::Perpendicular(writer) => writer.write::<P, I>(local_index, accumulator, inst),
        }
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        match self {
            Writer::Parallel(writer) => writer.commit_required(),
            Writer::Perpendicular(writer) => writer.commit_required(),
        }
    }

    pub fn commit(&mut self) {
        match self {
            Writer::Parallel(writer) => writer.commit(),
            Writer::Perpendicular(writer) => writer.commit(),
        }
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        match self {
            Writer::Parallel(writer) => writer.write_count(),
            Writer::Perpendicular(writer) => writer.write_count(),
        }
    }
}

#[derive(CubeType)]
pub struct ParallelWriter<Out: NumericVector> {
    output: View<Vector<Out::T, Out::N>, Coords2d, ReadWrite>,
    buffer: Value<Vector<Out::T, Out::N>>,
    axis_size: usize,
    write_index: usize,
    #[cube(comptime)]
    accumulator_length: usize,
}

#[cube]
impl<Out: NumericVector> ParallelWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] accumulator_format: AccumulatorFormat,
    ) -> ParallelWriter<Out> {
        let layout = build_reduce_output_layout::<Out>(
            output,
            reduce_axis,
            out_vec_axis,
            accumulator_format.len(),
        );

        ParallelWriter::<Out> {
            output: output.view_mut(layout),
            buffer: match accumulator_format {
                AccumulatorFormat::Single => Value::new_single(Vector::empty()),
                AccumulatorFormat::Multiple(length) => Value::new_Multiple(Array::new(length)),
            },
            axis_size: input.shape(reduce_axis),
            write_index,
            accumulator_length: accumulator_format.len(),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        let out = I::to_output_parallel::<Out::T>(inst, accumulator, self.axis_size);

        match out {
            Value::Multiple(array) =>
            {
                #[unroll]
                for i in 0..self.accumulator_length {
                    let mut vec = self.buffer.multiple_mut()[i];
                    vec[local_index] = array[i];
                    self.buffer.multiple_mut()[i] = vec;
                }
            }
            Value::Single(element) => {
                self.buffer.item()[local_index] = element.unwrap();
            }
            Value::None => {
                unreachable!()
            }
        }
    }

    pub fn commit(&mut self) {
        match &mut self.buffer {
            Value::Multiple(array) => {
                let write_index = self.write_index as u32;
                #[unroll]
                for k_iter in 0..self.accumulator_length {
                    let k_u32 = comptime!(k_iter as u32);
                    self.output
                        .write((write_index, k_u32.runtime()), array[k_iter])
                }
            }
            Value::Single(vector) => self
                .output
                .write((self.write_index as u32, 0), vector.unwrap()),
            Value::None => unreachable!(),
        }
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        match &self.buffer {
            Value::Multiple(array) => array[0].vector_size(),
            Value::Single(vector) => vector.unwrap().vector_size(),
            Value::None => unreachable!(),
        }
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        true
    }
}

#[derive(CubeType)]
pub struct PerpendicularWriter<Out: NumericVector> {
    output: View<Vector<Out::T, Out::N>, Coords2d, ReadWrite>,
    axis_size: usize,
    #[cube(comptime)]
    input_vector_size: VectorSize,
    #[cube(comptime)]
    output_vector_size: VectorSize,
    write_index: usize,
    #[cube(comptime)]
    accumulator_length: usize,
}

#[cube]
impl<Out: NumericVector> PerpendicularWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] accumulator_format: AccumulatorFormat,
    ) -> PerpendicularWriter<Out> {
        let input_vector_size = input.vector_size();
        let output_vector_size = output.vector_size();

        let layout = build_reduce_output_layout::<Out>(
            output,
            reduce_axis,
            out_vec_axis,
            accumulator_format.len(),
        );

        PerpendicularWriter::<Out> {
            output: output.view_mut(layout),
            axis_size: input.shape(reduce_axis),
            write_index,
            input_vector_size,
            output_vector_size,
            accumulator_length: accumulator_format.len(),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        _local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        let out = I::to_output_perpendicular::<Out::T>(inst, accumulator, self.axis_size);

        match out {
            Value::Multiple(array) => self.write_multiple::<P::SI>(array),
            Value::Single(vector) => self.write_single::<P::SI>(vector.unwrap(), 0),
            Value::None => unreachable!(),
        }
    }

    pub fn commit(&mut self) {
        // Nothing to do.
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        1
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        false
    }
}

#[cube]
impl<Out: NumericVector> PerpendicularWriter<Out> {
    fn write_single<S: Size>(&self, vector: Vector<Out::T, S>, k_index: usize) {
        if comptime![self.output_vector_size == self.input_vector_size] {
            self.output.write(
                (self.write_index as u32, k_index as u32),
                Vector::cast_from(vector),
            );
        } else {
            let num_iters = comptime![self.input_vector_size / self.output_vector_size];

            #[unroll]
            for i in 0..num_iters {
                let mut tmp = Vector::empty();

                #[unroll]
                for j in 0..self.output_vector_size {
                    tmp[j] = Out::T::cast_from(vector[i * self.output_vector_size + j]);
                }

                let index = self.write_index * num_iters + i;
                self.output.write((index as u32, k_index as u32), tmp);
            }
        }
    }

    fn write_multiple<S: Size>(&self, array: Array<Vector<Out::T, S>>) {
        #[unroll]
        for i in 0..self.accumulator_length {
            self.write_single(array[i], i);
        }
    }
}

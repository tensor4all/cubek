use std::marker::PhantomData;

use cubecl::comptime;
use cubecl::cube;
use cubecl::frontend::CubeIndexMutExpand;
use cubecl::prelude::*;

use crate::components::instructions::AccumulatorFormat;
use crate::components::instructions::{Accumulator, Item, Value};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{ReduceRequirements, ReduceStep, SharedAccumulator},
};
use cubecl::frontend::Numeric;

#[derive(Debug, CubeType, Clone)]
pub struct TopK {
    #[cube(comptime)]
    pub k: usize,
}

impl ReduceFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = usize;
}

#[derive(CubeType)]
pub struct TopkAccumulator<E: Scalar, S: Size> {
    pub elements: Array<Vector<E, S>>,
    pub coordinates: Array<Vector<u32, S>>,
}

#[derive(CubeType)]
/// Only to respect the type system. Shared Accumulator behaviour is not supported
pub struct DummyTopkSharedAccumulator<A: CubeType + Send + Sync + 'static> {
    #[cube(comptime)]
    _phantom: PhantomData<A>,
}

#[cube]
impl<A: CubeType + Send + Sync + 'static, P: ReducePrecision> SharedAccumulator<P>
    for DummyTopkSharedAccumulator<A>
{
    fn allocate(#[comptime] _length: usize, #[comptime] _coordinate: bool) -> Self {
        unreachable!()
    }

    fn read(_accumulator: &Self, _index: usize) -> Accumulator<P> {
        unreachable!()
    }

    fn write(_accumulator: &mut Self, _index: usize, _item: Accumulator<P>) {
        unreachable!()
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = DummyTopkSharedAccumulator<Accumulator<P>>;
    type Config = usize;

    fn requirements(_this: &Self) -> super::ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        TopK { k: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let mut elements = Array::new(comptime!(this.k));
        for i in 0..this.k {
            elements[i] = Vector::new(P::EA::min_value());
        }

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args: Value::new_None(),
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let item = item.elements;

        let candidate_item = match reduce_step {
            ReduceStep::Plane => {
                todo!()
            }
            ReduceStep::Identity => item,
        };

        let elements = accumulator.elements.multiple_mut();
        let mut item = Vector::cast_from(candidate_item);

        for k_iter in 0..this.k {
            let current_item = elements[k_iter];

            let keep = current_item.greater_than(item);
            let new_top_item = select_many(keep, current_item, item);
            let new_rest_item = select_many(keep, item, current_item);

            elements[k_iter] = new_top_item;
            item = new_rest_item;
        }
    }

    fn plane_reduce_inplace(_this: &Self, _accumulator: &mut Accumulator<P>) {
        todo!()
    }

    fn fuse_accumulators(_this: &Self, _accumulator: &mut Accumulator<P>, _other: &Accumulator<P>) {
        todo!("fuse_accumulator Not implemented")
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let accumulators = accumulator.elements.multiple();
        let vector_size = accumulators[0].size().comptime();

        let mut topk = Array::new(this.k);

        for i in 0..this.k {
            for j in 0..vector_size {
                let mut element = Out::cast_from(accumulators[i][j]);

                for slot in 0..this.k {
                    let current = topk[slot];

                    let keep = current > element;

                    topk[slot] = select(keep, current, element);
                    element = select(keep, element, current);
                }
            }
        }

        Value::new_Multiple(topk)
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        // TODO if Out==P::EA, return acc_values directly

        let acc_values = accumulator.elements.multiple();
        let mut output = Array::new(this.k);

        for i in 0..this.k {
            output[i] = Vector::cast_from(acc_values[i]);
        }

        Value::new_Multiple(output)
    }
}

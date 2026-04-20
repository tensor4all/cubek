use std::marker::PhantomData;

use cubecl::comptime;
use cubecl::cube;
use cubecl::frontend::CubeIndexMutExpand;
use cubecl::prelude::*;

use crate::components::instructions::{Accumulator, AccumulatorKind, Item};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{ReduceRequirements, ReduceStep, SharedAccumulator},
};
use cubecl::frontend::Numeric;

#[derive(Debug, CubeType, Clone)]
pub struct ArgTopK {
    #[cube(comptime)]
    pub k: u32,
}

impl ReduceFamily for ArgTopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = u32;
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
impl<P: ReducePrecision> ReduceInstruction<P> for ArgTopK {
    type SharedAccumulator = DummyTopkSharedAccumulator<Accumulator<P>>;
    type Config = u32;

    fn requirements(_this: &Self) -> super::ReduceRequirements {
        ReduceRequirements { coordinates: true }
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        ArgTopK { k: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let mut elements = Array::new(comptime!(this.k as usize));
        let mut args = Array::new(comptime!(this.k as usize));
        for i in 0..this.k {
            elements[i as usize] = Vector::new(P::EA::min_value());
            args[i as usize] = Vector::new(u32::MAX);
        }

        Accumulator::<P> {
            elements: AccumulatorKind::new_Multiple(elements),
            args: AccumulatorKind::new_Multiple(args),
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        todo!()

        // let coordinate = item.args.item();
        // let item = item.elements;

        // let (candidate_item, candidate_coordinate) = match reduce_step {
        //     ReduceStep::Plane => {
        //         todo!()
        //         // let candidate_item = plane_max(item);
        //         // let candidate_coordinate =
        //         //     lowest_coordinate_matching(candidate_item, item, coordinate);
        //         // (candidate_item, candidate_coordinate)
        //     }
        //     ReduceStep::Identity => (item, coordinate),
        // };

        // let (elements, args) = accumulator.to_elements_and_args();
        // let elements = elements.multiple();
        // let args = args.multiple();
        // let mut item = Vector::cast_from(item);

        // for k_iter in 0..this.k {
        //     // let current = elements[k_iter as usize];
        //     // elements[k_iter as usize] = max(current, item);
        //     // item = min(current, item);

        //     let current_item = elements[k_iter as usize];
        //     let current_coord = args[k_iter as usize];

        //     // Reuse your existing tie-breaking logic:
        //     // keep "0" means items[0] wins the top slot
        //     let keep0 = select_many(
        //         current_item.equal(item),
        //         current_coord.less_than(coordinate),
        //         current_item.greater_than(item),
        //     );

        //     let new_top_item = select_many(keep0, current_item, item);
        //     let new_top_coord = select_many(keep0, current_coord, coordinate);
        //     let new_rest_item = select_many(keep0, item, current_item);
        //     let new_rest_coord = select_many(keep0, coordinate, current_coord);

        //     elements[k_iter as usize] = new_top_item;
        //     args[k_iter as usize] = new_top_coord;
        //     item = new_rest_item;
        //     coordinate = new_rest_coord;
        // }

        // Accumulator::<P> {
        //     elements: AccumulatorKind::new_Multiple(elements),
        //     args: AccumulatorKind::new_Multiple(args),
        // }
    }

    fn plane_reduce_inplace(_this: &Self, _accumulator: &mut Accumulator<P>) {
        todo!()
    }

    fn fuse_accumulators(_this: &Self, _accumulator: &mut Accumulator<P>, _other: &Accumulator<P>) {
        todo!("fuse_accumulator Not implemented")
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        _accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        todo!("merge_vector Not implemented")
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        _accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        todo!("to_output_perpendicular Not implemented")
    }
}

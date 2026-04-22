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
        //todo!();
        let elements = accumulator.elements.multiple_mut();

        match reduce_step {
            ReduceStep::Plane => {
                // Every thread starts with its own item as the candidate
                let mut local_best_val = Vector::cast_from(item.elements);
                let unit_pos_plane = Vector::new(UNIT_POS_X);

                for _i in 0..this.k {
                    // 1. Find the global maximum among currently unmasked items
                    let winning_val = plane_max(local_best_val);

                    // 2. Tie-break: Determine which specific lane owns this value
                    let is_match = local_best_val.equal(winning_val);
                    let my_claim = select_many(is_match, unit_pos_plane, Vector::new(u32::MAX));
                    let winning_lane = plane_min(my_claim);

                    // 3. Insert the global winner into the local Top-K set
                    // (All threads in the warp insert the same value to stay in sync)
                    let mut insert_item = winning_val;
                    for j in 0..this.k {
                        let acc_item = elements[j];
                        let keep = acc_item.greater_than(insert_item);

                        elements[j] = select_many(keep, acc_item, insert_item);
                        insert_item = select_many(keep, insert_item, acc_item);
                    }

                    // 4. Mask out the winner: the thread that just "gave" its value
                    // sets its candidate to MIN so it's ignored in the next k_winners loop.
                    let is_winner_thread = unit_pos_plane.equal(winning_lane);
                    local_best_val = select_many(
                        is_winner_thread,
                        Vector::new(P::EA::min_value()),
                        local_best_val,
                    );
                }
            }
            ReduceStep::Identity => {
                // Each thread just inserts its single item into its own Top-K list
                let mut insert_item = Vector::cast_from(item.elements);

                for j in 0..this.k {
                    let acc_item = elements[j];
                    let keep = acc_item.greater_than(insert_item);

                    elements[j] = select_many(keep, acc_item, insert_item);
                    insert_item = select_many(keep, insert_item, acc_item);
                }
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        //todo!();
        let elements = accumulator.elements.multiple_mut();

        // We only need to store the final elements
        let mut final_elements = Array::new(this.k);

        // 'local_ptr' tracks which of the K items in our local list we are proposing.
        let mut local_ptr = Vector::new(0u32);
        // 'lane_id' gives every thread a unique ID for tie-breaking.
        let lane_id = Vector::new(UNIT_POS_X);

        for i in 0..this.k {
            // 1. Fetch the current local candidate
            let mut local_best_val = Vector::new(P::EA::min_value());
            for j in 0..this.k {
                let is_pointed_slot = local_ptr.equal(Vector::new(j as u32));
                local_best_val = select_many(is_pointed_slot, elements[j], local_best_val);
            }

            // 2. Find the global max value
            let winning_val = plane_max(local_best_val);

            // 3. TIE-BREAKER: Find WHICH thread provided the winner.
            let is_candidate = local_best_val.equal(winning_val);
            let candidate_id = select_many(is_candidate, lane_id, Vector::new(u32::MAX));
            let winning_lane_id = plane_min(candidate_id);

            // 4. Record the winner
            final_elements[i] = winning_val;

            // 5. Update pointer: Only the specific thread (and lane) that won increments.
            let is_winner_thread = lane_id.equal(winning_lane_id);
            local_ptr = select_many(is_winner_thread, local_ptr + Vector::new(1u32), local_ptr);
        }

        for i in 0..this.k {
            elements[i] = final_elements[i];
        }
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let acc_elements = accumulator.elements.multiple_mut();
        let other_elements = other.elements.multiple();

        for i in 0..this.k {
            let mut item = other_elements[i];
            for j in 0..this.k {
                let current_item = acc_elements[j];
                let keep = current_item.greater_than(item);

                let new_top_item = select_many(keep, current_item, item);
                let new_rest_item = select_many(keep, item, current_item);

                acc_elements[j] = new_top_item;
                item = new_rest_item;
            }
        }
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

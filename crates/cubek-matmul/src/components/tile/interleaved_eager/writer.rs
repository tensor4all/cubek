use cubecl::prelude::*;

use crate::components::tile::{
    StridedTile, TileConfig as _,
    interleaved_eager::{InterleavedEagerAccumulator, config::InterleavedEagerMatmulConfig},
};

/// Writer for the interleaved matmul fragments.
///
/// Before writing, sums all the unit accumulators
#[derive(CubeType)]
pub struct InterleavedStageWriter {}

#[cube]
impl InterleavedStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &InterleavedEagerAccumulator<A>,
        #[comptime] config: InterleavedEagerMatmulConfig,
    ) {
        assert!(
            tile.stage.line_size().comptime() == 1,
            "out stage line size should be 1, got {:?}",
            tile.stage.line_size().comptime()
        );

        #[unroll]
        for i in 0..config.num_local_accumulators() {
            let index = i as u32 * config.plane_dim() + UNIT_POS_X;
            let offs = tile.stage_offset(index);
            let elem = acc.array[i];
            // if UNIT_POS_X != 0 {
            tile.stage[offs as usize] = Line::cast_from(elem);
            // } else {
            //     tile.stage[offs as usize] = Line::cast_from(0);
            // }
        }
    }
}

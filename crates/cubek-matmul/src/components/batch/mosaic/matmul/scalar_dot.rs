use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    mosaic::io::{read, write},
};

/// Placeholder Mosaic kernel: one output cell per plane, scalar dot
/// product along K. Exists only so the boilerplate (family + routine +
/// strategy + tests + bench) is exercisable end-to-end. The next step is
/// to replace this with a tile-API version where the accumulator type
/// drives the structure of the inner loop.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_scalar_dot<
    L: CubePrimitive,
    R: CubePrimitive,
    O: CubePrimitive,
    AccR: Numeric,
    N: Size,
>(
    lhs: View<L, Coords2d>,
    rhs: View<R, Coords2d>,
    out: View<O, Coords2d, ReadWrite>,
    m_pos: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;
    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let k_pos = tile_index * vector_size;
        let lhs_val = read(&lhs, (m_pos, k_pos), check_bounds);
        let rhs_val = read(&rhs, (k_pos, n_pos), check_bounds);
        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    let sum = O::cast_from(Vector::vector_sum(acc));
    write(&out, (m_pos, n_pos), sum, check_bounds);
}

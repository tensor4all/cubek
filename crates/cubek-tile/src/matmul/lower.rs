//! Lowering `c.mma(a, b)`: a tile with levels left lowers per its [`Schedule`]; a final tile
//! contracts via the [`Mma`](super::leaf::Mma) leaf.

use cubecl::prelude::*;

use super::leaf::Mma;
use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::*;

#[cube]
impl<Acc: CubePrimitive> Tile<Acc> {
    /// `c.mma(a, b)`: a tile with levels left lowers per its [`Schedule`]; a final tile
    /// contracts via [`Mma`].
    pub fn mma<Lhs: CubePrimitive, Rhs: CubePrimitive>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>)
    where
        Acc: Mma<Lhs, Rhs>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => Acc::mma(self, lhs, rhs),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => mma_direct(lhs, rhs, self),
                Schedule::Staged => mma_staged(lhs, rhs, self),
                Schedule::DoubleBuffered => mma_double(lhs, rhs, self),
            },
        }
    }

    /// The [`Direct`](Schedule::Direct) lowering's per-region step.
    pub fn mma_at<Lhs: CubePrimitive, Rhs: CubePrimitive>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        region: &Region,
    ) where
        Acc: Mma<Lhs, Rhs>,
    {
        self.at(region).mma(&lhs.at(region), &rhs.at(region));
    }
}

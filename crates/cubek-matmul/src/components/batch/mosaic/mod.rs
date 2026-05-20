//! CPU matmul sandbox built around a tile-centric API:
//! `partition` reshapes a 2D view into a 4D tile grid, `copy` is always
//! lazy, and `tile_acc.mma(lhs_tile, rhs_tile)` drives the math. The
//! accumulator choice is what drives the rest of the algorithm —
//! everything else is intentionally minimal so the API can evolve.

mod config;
pub(crate) mod io;
mod mat_layout;
mod matmul;
mod setup;

pub use config::*;
pub use matmul::*;
pub use setup::*;

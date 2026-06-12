//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`. A final tile
//! contracts via the [`Mma`](leaf::Mma) leaf; otherwise [`mma`](Tile::mma) lowers (partition,
//! locate each operand, recurse) per the head [`Schedule`]. The leaf lives in [`leaf`] (with
//! the memory and tensor-core paths in [`register`] and [`cmma`]); the lowering in [`lower`],
//! the schedules in [`schedule`].

mod cmma;
mod leaf;
mod lower;
mod register;
mod schedule;

use std::marker::PhantomData;

use cubecl::{cmma::Matrix, ir::MatrixLayout, prelude::*, std::tensor::layout::Coords2d};

pub struct Unit;
pub struct Plane;
pub struct Cube;

#[derive(CubeType)]
pub struct Tile<E: Scalar, N: Size, S = Unit, IO: SliceVisibility = ReadOnly> {
    pub storage: TileStorage<E, N, IO>,
    pub layout: TileLayout,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[derive(CubeType)]
pub struct TileStorage<E: Scalar, N: Size, IO: SliceVisibility = ReadOnly> {
    kind: TileStorageKind<E, N, IO>,
}

#[derive(CubeType)]
enum TileStorageKind<E: Scalar, N: Size, IO: SliceVisibility = ReadOnly> {
    GlobalMemory(Slice<Vector<E, N>, IO>),
    SharedMemory(Slice<Vector<E, N>, IO>),
    LocalMemory(Slice<Vector<E, N>, IO>),
    Cmma(Matrix<E>),
    Mma(Array<Vector<E, N>>),
    Broadcasted(E),
}

#[derive(CubeType)]
pub enum TileLayout {
    Contiguous(MatrixLayout),
    Strided(StridedLayout),
}

#[derive(CubeType)]
pub struct StridedLayout {
    pub strides: Coords2d,
    pub shape: Coords2d,
}

#[cube]
impl<E: Scalar, N: Size, IO: SliceVisibility> TileStorage<E, N, IO> {
    pub fn as_slice(&self) -> Slice<Vector<E, N>, IO> {
        match &self.kind {
            TileStorageKind::GlobalMemory(slice) => slice.clone(),
            TileStorageKind::SharedMemory(slice) => slice.clone(),
            TileStorageKind::LocalMemory(slice) => slice.clone(),
            TileStorageKind::Cmma(_) => panic!(),
            TileStorageKind::Mma(_) => panic!(),
            TileStorageKind::Broadcasted(_) => panic!(),
        }
    }
}

// TODO
//
// impl<E: Scalar, N: Size, IO: SliceVisibility> Tile<E, N, Cube, IO> {
//     pub fn partition(self, strategy: Cube2PlaneParitioning) -> Sequence<Tile<E, N, Plane, IO>> {
//         todo!()
//     }
// }

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Coords2d, Layout, LayoutExpand},
};

#[derive(CubeType, Clone)]
pub struct ReduceOutputLayout {
    num_vectored_reductions: usize,
    accumulator_length: usize,
}

#[cube]
impl ReduceOutputLayout {
    pub fn new(num_vectored_reductions: usize, accumulator_length: usize) -> ReduceOutputLayout {
        ReduceOutputLayout {
            num_vectored_reductions,
            accumulator_length,
        }
    }
}

#[cube]
impl Layout for ReduceOutputLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords1d {
        let write_index = coords.0 as usize;
        let k_iter = coords.1 as usize;
        k_iter * self.num_vectored_reductions + write_index
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords1d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (
            self.num_vectored_reductions as u32,
            self.accumulator_length as u32,
        )
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos.0 < self.num_vectored_reductions as u32 && pos.1 < self.accumulator_length as u32
    }
}

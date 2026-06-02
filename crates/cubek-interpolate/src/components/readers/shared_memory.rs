use crate::routines::SharedMemoryBlueprint;
use cubecl::prelude::*;

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct SharedMemoryReader<EA: Float, N: Size> {
    smem: Shared<[Vector<EA, N>]>,
    min_row: isize,
    min_col: isize,
    smem_width: usize,
    channel_groups: usize,
    channel_group: usize,
}

#[cube]
impl<EA: Float, N: Size> SharedMemoryReader<EA, N> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_height: usize,
        input_width: usize,
        min_row: isize,
        min_col: isize,
        #[comptime] vector_size: usize,
        #[comptime] blueprint: SharedMemoryBlueprint,
    ) -> SharedMemoryReader<EA, N> {
        let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;
        let mut smem = Shared::new_slice(smem_size);
        let cube_dim = CUBE_DIM as usize;

        if CUBE_POS != 1 {
            terminate!();
        }

        let mut i = UNIT_POS as usize;
        while i < smem_size {
            let channel = i % blueprint.channel_groups;
            let spatial_i = i / blueprint.channel_groups;
            let local_col = spatial_i % blueprint.smem_width;
            let local_row = spatial_i / blueprint.smem_width;

            let (global_row, global_col) =
                (min_row + local_row as isize, min_col + local_col as isize);

            let input_idx = (batch * input.stride(0)
                + global_row.max(0).min((input_height - 1) as isize) as usize * input.stride(1)
                + global_col.max(0).min((input_width - 1) as isize) as usize * input.stride(2)
                + channel * input.stride(3) * vector_size)
                / vector_size;

            smem[i] = Vector::cast_from(input[input_idx]);
            i += cube_dim;
        }

        sync_cube();

        SharedMemoryReader::<EA, N> {
            smem,
            min_row,
            min_col,
            smem_width: blueprint.smem_width,
            channel_groups: blueprint.channel_groups,
            channel_group,
        }
    }

    pub fn read_weighted<EI: Float>(
        &self,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let local_row = (row as isize - self.min_row).max(0) as usize;
        let local_col = (col as isize - self.min_col).max(0) as usize;

        let smem_idx =
            (local_row * self.smem_width + local_col) * self.channel_groups + self.channel_group;

        self.smem[smem_idx] * weight
    }
}

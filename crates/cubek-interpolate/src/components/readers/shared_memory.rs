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
    channel: usize,
}

/// Gathers the input element that belongs in shared-memory slot `i`.
///
/// Slot `i` is a flat index into the `[smem_height, smem_width, channel_groups]`
/// tile (in vector units). This recovers the `(row, col, channel_group)` it maps
/// to, clamps the global coordinates to the input edges, and returns the loaded
/// vector. Shared verbatim by the production reader and the isolated loading test.
#[cube]
pub fn smem_slot_value<EI: Float, EA: Float, N: Size>(
    input: &Tensor<Vector<EI, N>>,
    i: usize,
    batch: usize,
    input_height: usize,
    input_width: usize,
    min_row: isize,
    min_col: isize,
    #[comptime] vector_size: usize,
    #[comptime] blueprint: SharedMemoryBlueprint,
) -> Vector<EA, N> {
    let local_channel = i % blueprint.channel_groups;
    let local_pos = i / blueprint.channel_groups;
    let local_col = local_pos % blueprint.smem_width;
    let local_row = local_pos / blueprint.smem_width;

    let (global_row, global_col) = (min_row + local_row as isize, min_col + local_col as isize);

    let input_idx = (batch * input.stride(0)
        + global_row.max(0).min((input_height - 1) as isize) as usize * input.stride(1)
        + global_col.max(0).min((input_width - 1) as isize) as usize * input.stride(2)
        + local_channel * input.stride(3) * vector_size)
        / vector_size;

    Vector::cast_from(input[input_idx])
}

/// Fills `dst` with the shared-memory tile region, spreading the slots across the
/// cube's units. `dst` is indexed in vector units exactly like the production
/// shared-memory buffer, so the same code drives both the kernel (writing into a
/// `Shared` slice) and the loading test (writing into an output `Tensor`).
#[cube]
pub fn load_shared_region<EI: Float, EA: Float, N: Size, L: List<Vector<EA, N>> + ?Sized>(
    input: &Tensor<Vector<EI, N>>,
    dst: &mut L,
    batch: usize,
    input_height: usize,
    input_width: usize,
    min_row: isize,
    min_col: isize,
    #[comptime] vector_size: usize,
    #[comptime] blueprint: SharedMemoryBlueprint,
) where
    // Lets the index-assign into the generic `L` resolve `Vector`'s expand type,
    // which the compiler won't normalize on its own in a generic context.
    Vector<EA, N>: CubeType<ExpandType = NativeExpand<Vector<EA, N>>>,
{
    let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;

    let unit_pos = UNIT_POS as usize;
    let cube_dim = CUBE_DIM as usize;

    // Units beyond the region have nothing to load; guard against the unsigned
    // `smem_size - unit_pos` underflowing when there are more units than slots.
    let num_iterations = if unit_pos < smem_size {
        (smem_size - unit_pos).div_ceil(cube_dim)
    } else {
        0usize
    };

    for step in 0..num_iterations {
        let i = unit_pos + step * cube_dim;

        dst[i] = smem_slot_value::<EI, EA, N>(
            input,
            i,
            batch,
            input_height,
            input_width,
            min_row,
            min_col,
            vector_size,
            blueprint,
        );
    }
}

#[cube]
impl<EA: Float, N: Size> SharedMemoryReader<EA, N> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel: usize,
        input_height: usize,
        input_width: usize,
        min_row: isize,
        min_col: isize,
        #[comptime] vector_size: usize,
        #[comptime] blueprint: SharedMemoryBlueprint,
    ) -> SharedMemoryReader<EA, N> {
        let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;
        let mut smem = Shared::new_slice(smem_size);

        load_shared_region::<EI, EA, N, Shared<[Vector<EA, N>]>>(
            input,
            &mut smem,
            batch,
            input_height,
            input_width,
            min_row,
            min_col,
            vector_size,
            blueprint,
        );

        sync_cube();

        SharedMemoryReader::<EA, N> {
            smem,
            min_row,
            min_col,
            smem_width: blueprint.smem_width,
            channel_groups: blueprint.channel_groups,
            channel,
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

        let smem_idx = (local_row * self.smem_width + local_col) * self.channel_groups + self.channel;

        self.smem[smem_idx] * weight
    }
}

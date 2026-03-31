use std::marker::PhantomData;

use crate::components::batch::base::BatchMatmulFamily;
use crate::components::batch::no_stage_vecmat::{
    NoStageVecMatBlueprint, NoStageVecMatConfig, NoStageVecMatFamily,
};
use crate::components::batch::{BatchConfig as _, SliceIndex};

use crate::{components::batch::BatchMatmul, definition::*, launch::MatmulArgs};
use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero};

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
#[allow(clippy::type_complexity)]
/// Launches the matmul kernel
pub(crate) fn matmul_entry<
    Args: MatmulArgs<Config = ()>,
    Lhs: Numeric,
    LhsSize: Size,
    Rhs: Numeric,
    RhsSize: Size,
    Acc: Numeric,
    AccSize: Size,
>(
    inputs: &<Args as MatmulArgs>::Input<
        Vector<Lhs, LhsSize>,
        Vector<Rhs, RhsSize>,
        Vector<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Vector<Acc, AccSize>>,
    runtime_config: (),
    cube_mapping: CubeMapping,
    #[comptime] blueprint: NoStageVecMatBlueprint,
    #[define(Lhs, Rhs, Acc)] _global: [StorageType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let mut state =
        Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
            inputs,
            output,
            runtime_config,
            blueprint.lhs_global_layout_config(),
            blueprint.rhs_global_layout_config(),
            blueprint.out_global_layout_config(),
        );

    let vector_size_lhs = Args::view_lhs(&state).vector_size();
    let vector_size_rhs = Args::view_rhs(&state).vector_size();
    let vector_size_out = Args::view_out(&mut state).vector_size();
    let vector_sizes = comptime!(MatmulVectorSizes {
        lhs: vector_size_lhs,
        rhs: vector_size_rhs,
        out: vector_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(NoStageVecMatFamily::expand_config(
        &device_props,
        &blueprint,
        &blueprint.dtypes,
        &vector_sizes
    ));

    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = comptime!(config.unwrap());

    let mut state =
        Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
            inputs,
            output,
            runtime_config,
            config.lhs_global_layout_config(),
            config.rhs_global_layout_config(),
            config.out_global_layout_config(),
        );

    let define!(RegisterLhs) = blueprint.dtypes.lhs_register;
    let define!(RegisterRhs) = blueprint.dtypes.rhs_register;
    let define!(RegisterAcc) = blueprint.dtypes.acc_register;

    NoStageVecMat::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs),
        (Acc, AccSize, Acc, AccSize, RegisterAcc),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct NoStageVecMat<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for NoStageVecMat<MP> {
    type Config = NoStageVecMatConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let num_planes = config.num_planes;
        let plane_dim = config.plane_dim;

        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, _, n) = out.shape();
        let (_, n_cube_id, batch_cube_id) = cube_mapping.cube_pos_to_tensor_pos();

        let lhs_batch = Args::batch_lhs(state, batch_cube_id as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube_id as usize);
        let out_batch = Args::batch_out(state, batch_cube_id as usize);

        let lhs = lhs.view(SliceIndex::new(lhs_batch, lhs.shape()));
        let rhs = rhs.view(SliceIndex::new(rhs_batch, rhs.shape()));
        let out = out.view_mut(SliceIndex::new(out_batch, out.shape()));

        let size!(NA) = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let vector_size = NA::value() as u32;

        let plane_id = UNIT_POS_Y;
        let unit_id = UNIT_POS_X;

        // First, we load a 32*8 segment of lhs
        // Then a 32*8 column segment of rhs
        // Then it's simply an element wise multiplication
        // Yielding 32*8 accumulators of 1 element
        // Which at the end are FIRST vector summed, then scalar plane summed
        // This is for one k
        // Then we can send one plane per k

        let absolute_plane_id = n_cube_id * num_planes + plane_id;

        // Tile = 1d vector of plane_dim * vector_size
        let tile_size = plane_dim * vector_size;
        let num_tiles = k / tile_size;

        let mut acc = Vector::<AccR<MP>, NA>::zero();

        for tile_index in 0..num_tiles {
            let swizzled_tile_index = (tile_index + plane_id) % num_tiles;
            let k_base = swizzled_tile_index * plane_dim;

            let lhs_vec = lhs.read_checked((0, (k_base + unit_id) * vector_size));
            let rhs_vec = rhs.read_checked(((k_base + unit_id) * vector_size, absolute_plane_id));

            acc += Vector::cast_from(lhs_vec) * Vector::cast_from(rhs_vec);
        }

        let mut sum = AccR::<MP>::zero();

        #[unroll]
        for i in 0..NA::value() {
            sum += acc[i];
        }

        let sum = plane_sum(sum);

        if unit_id == 0 {
            out.write_checked((0, absolute_plane_id), Vector::cast_from(sum));
        }
    }
}

#[cube]
fn shuffle<E: Numeric, N: Size>(
    shared_value: Vector<E, N>,
    unit: u32,
    #[comptime] plane_dim: u32,
) -> Vector<E, N> {
    if comptime!(plane_dim > 1) {
        plane_shuffle(shared_value, unit)
    } else {
        shared_value
    }
}

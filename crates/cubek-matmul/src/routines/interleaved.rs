use cubecl::features::MmaConfig;
use cubecl::{Runtime, client::ComputeClient};
use std::fmt::Display;
use std::marker::PhantomData;

use crate::components::batch::BatchMatmulFamily;
use crate::components::tile::interleaved_deferred::InterleavedDeferredMatmul;
use crate::components::tile::interleaved_eager::InterleavedEagerMatmul;
use crate::components::tile::io::{Filled, Strided};
use crate::definition::{
    CubeCountStrategy, GlobalOrderStrategy, HypercubeBlueprint, MatmulElems, MatmulLineSizes,
    MatmulProblem, MatmulSetupError, MultiRowStrategy, SmAllocation, TilingBlueprint, TilingScheme,
    adjust_dtypes,
};
use crate::routines::{BlueprintStrategy, DeviceSettings, LaunchInfo};
use crate::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, PartitionBuffering, PlaneMatmulFamily,
            RowMajorTilingOrder, StridedStageFamily,
        },
        tile::TileMatmulFamily,
    },
    routines::{
        Routine,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    },
};

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct InterleavedAlgorithm<
    TMM: TileMatmulFamily,
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

#[derive(Default, Debug, Clone)]
pub struct InterleavedArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl Display for InterleavedArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.multi_rows { "_multi_rows" } else { "" })
    }
}

impl<TMM: TileMatmulFamily, LL, RL> Routine for InterleavedAlgorithm<TMM, LL, RL>
where
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    TMM:
        TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled, OutTile = Strided>,
{
    type Strategy = InterleavedArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SimpleMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            LL,
            RL,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if InterleavedDeferredMatmul::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let client = &device_settings.client;
        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => {
                if strategy.multi_rows {
                    infer_blueprint_multi_rows::<R, InterleavedDeferredMatmul>(
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.line_sizes,
                    )
                } else {
                    infer_blueprint_plane::<InterleavedDeferredMatmul, R>(
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.line_sizes,
                        PlaneTilingBlueprintOptions {
                            partition_buffering: Some(PartitionBuffering::Single),
                            tiny_selection_enabled: true,
                            swizzled: InterleavedDeferredMatmul::should_swizzle(client),
                            ..Default::default()
                        },
                    )
                }?
            }
        };

        Self::validate_blueprint(
            client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

fn infer_blueprint_multi_rows<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> Result<(TilingBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, TMM::requires_accelerator());

    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::Flattened,
    };

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((4, 4, 32).into())
        .with_partition_size((1, 1, 1).into())
        .with_stage_size((4, 4, 1).into())
        .build()
        .unwrap();

    let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
        .global_order_strategy(GlobalOrderStrategy::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
        .cube_count_strategy(cube_count_strategy)
        .build();

    Ok((
        TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_blueprint(hypercube)
            .build(),
        dtypes,
    ))
}

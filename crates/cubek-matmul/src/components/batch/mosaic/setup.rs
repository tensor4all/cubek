use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    ir::{AddressType, DeviceProperties},
    server::LaunchError,
};
use cubek_std::{MatrixLayout, cube_count::HypercubeBlueprint};

use crate::{
    components::{
        CubeDimResource,
        batch::{
            BatchMatmulFamily, CheckBounds,
            mosaic::{Mosaic, MosaicConfig, matmul::matmul_entry},
        },
        global::memory::GlobalLayoutConfig,
        stage::NumStages,
    },
    definition::{
        Blueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes,
        MatmulVectorSizes, SwizzleModes, TilingScheme,
    },
    launch::*,
};

/// CPU-only matmul family. Sandbox for iterating on a tile-centric API
/// (`partition`, lazy `copy`, `tile_acc.mma(...)`). Today it implements a
/// plain Row-Col scalar dot product per output cell; the goal is to grow
/// that into a tile-based formulation without touching the wiring.
pub struct MosaicFamily {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct MosaicBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub check_bounds: CheckBounds,
}

impl Blueprint for MosaicBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        panic!("Mosaic Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("Mosaic Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for MosaicFamily {
    type Matmul<MP: MatmulTypes> = Mosaic<MP>;
    type Config = MosaicConfig;
    type Blueprint = MosaicBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(MosaicConfig {
            plane_dim: device_props.hardware.plane_size_max,
            num_planes: blueprint.num_planes as u32,
            check_bounds: blueprint.check_bounds,
        })
    }

    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        cube_mapping: CubeMappingLaunch<R>,
        blueprint: MosaicBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                (),
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        Ok(CubeDimResource::Planes(blueprint.num_planes as u32))
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        _blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let plane_dim = client.properties().hardware.plane_size_max as usize;

        if plane_dim > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Mosaic is CPU-only (requires plane_dim == 1), got {}",
                plane_dim,
            ))));
        }

        if vector_sizes.lhs != vector_sizes.rhs {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs vector sizes must be equal, got lhs:{:?}, rhs:{:?}",
                vector_sizes.lhs, vector_sizes.rhs
            ))));
        }

        if vector_sizes.out != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Out vector size must be 1, got {:?}",
                vector_sizes.out,
            ))));
        }

        let vs = vector_sizes.lhs;
        if !problem.k.is_multiple_of(vs) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                problem.k, vs,
            ))));
        }

        // Sandbox start: only accept Row-Col (K-contig on both sides).
        // Other layout combinations belong to gemm for now.
        if !matches!(problem.lhs_layout, MatrixLayout::RowMajor)
            || !matches!(problem.rhs_layout, MatrixLayout::ColMajor)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Mosaic currently only supports Row-Col layouts, got lhs:{:?}, rhs:{:?}",
                problem.lhs_layout, problem.rhs_layout,
            ))));
        }

        Ok(())
    }
}

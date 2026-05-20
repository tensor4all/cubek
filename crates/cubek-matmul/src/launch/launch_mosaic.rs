use cubecl::{
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    definition::cube_mapping_launch,
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
};

use crate::{
    launch::InputArg,
    launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::mosaic::MosaicRoutine,
    routines::{BlueprintStrategy, Routine as _},
};

fn vector_size_for<R: Runtime>(
    client: &ComputeClient<R>,
    binding: &InputBinding<R>,
    default_size: usize,
    plane_size: usize,
    dim: usize,
) -> Result<usize, VectorizationError> {
    let (size, num_quants) = if let InputBinding::Quantized { scheme, .. } = binding {
        (scheme.size_bits_stored() / 8, scheme.num_quants())
    } else {
        (default_size, 1)
    };
    client
        .io_optimized_vector_sizes(size)
        .filter(|&v| dim.is_multiple_of(plane_size * v * num_quants))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), MosaicRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = rhs.shape().len();
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    let m = lhs_shape.to_vec()[rank - 2];
    let n = rhs_shape.to_vec()[rank - 1];
    let k = lhs_shape.to_vec()[rank - 1];

    let plane_size = client.properties().hardware.plane_size_max as usize;

    if !k.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Dimension k={} must be a multiple of plane size {}",
            k, plane_size
        ))));
    }

    let lhs_vector_size = vector_size_for(client, &lhs, dtypes.lhs_global.size(), plane_size, k)?;
    let rhs_vector_size = vector_size_for(client, &rhs, dtypes.rhs_global.size(), plane_size, k)?;

    let shared_vector_size = lhs_vector_size.min(rhs_vector_size);

    let vector_sizes = MatmulVectorSizes {
        lhs: shared_vector_size,
        rhs: shared_vector_size,
        out: 1,
    };

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let lhs_batches: Shape = lhs.shape().to_vec()[..rank - 2].into();
    let rhs_batches: Shape = rhs.shape().to_vec()[..rank - 2].into();

    let lhs_layout =
        MatrixLayout::from_shape_and_strides(lhs_shape, &lhs.data().strides, lhs.scheme())?;
    let rhs_layout =
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?;

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = MosaicRoutine::device_settings(client, vector_sizes);
    let expand_info = MosaicRoutine::expand_blueprint(&problem, &device_settings, strategy)?;
    let launch_info = MosaicRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<MosaicRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<MosaicRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );

    MosaicRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        (),
        cube_mapping_launch(&launch_info.cube_count_plan),
        launch_info.blueprint,
        dtypes,
        &launch_info.vector_sizes,
    )
}

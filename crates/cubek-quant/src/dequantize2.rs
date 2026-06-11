#![allow(missing_docs)] // pub cube modules

use cubecl::{calculate_cube_count_elemwise, ir::ElemType};
use cubecl::{features::TypeUsage, tensor_vector_size_parallel};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

use crate::{
    layout::{ScalesView, scales_view},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
    utils::packed_storage_elem,
};
use cubecl::std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scale: TensorBinding<R>,
    scheme: &QuantScheme,
    output_dtype: StorageType,
) -> Result<(), LaunchError> {
    let scale_dtype: StorageType = ElemType::from_quant_param(scheme.param).into();

    let axis = match scheme.store {
        QuantStore::PackedU32(axis) => axis,
        _ => todo!(),
    };

    let block_size = match scheme.level {
        QuantLevel::Block(block_size) => block_size.num_elements(),
        QuantLevel::Tensor => todo!(),
    };

    let vector_size_scale = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(scale_dtype.size()),
        &scale.shape,
        &scale.strides,
        axis,
    );

    let vector_size_store = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(core::mem::size_of::<u32>()),
        &input.shape,
        &input.strides,
        axis,
    );

    // A plane reads a single coalesced vector of scales, broadcasts them to its units, then streams
    // in the packed values as coalesced store rounds. The blueprint sizes that per-plane work.
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let _blueprint = calculate_blueprint(
        vector_size_store,
        vector_size_scale,
        scheme,
        block_size,
        plane_size,
    );

    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_kern<F: Float, NF: Size, FS: Numeric, QS: Int, NQ: Size, NS: Size>(
    input: LinearView<'_, Vector<QS, NQ>>,
    scales: ScalesView<'_, Vector<FS, NS>>,
    mut output: LinearViewMut<'_, Vector<F, NF>>,
    #[comptime] scheme: QuantScheme,
    #[comptime] blueprint: DequantizationBlueprint,
    #[define(F, FS, QS)] _dtypes: [StorageType; 3],
) {
}

/// The amount of work a single plane performs during dequantization, along with the vectorization
/// factors used to load the packed values and their scales.
///
/// The kernel is organized per-plane (not per-unit) so global memory stays coalesced: a contiguous
/// block of `block_size` packed values shares one scale, so the plane reads a single coalesced
/// vector of scales, broadcasts the right scale to each unit, then streams the packed values in as
/// coalesced rounds of store loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DequantizationBlueprint {
    /// Number of units in a plane.
    plane_size: usize,
    /// Number of packed store elements each unit loads together per round (store vectorization).
    vectorization_store: usize,
    /// Number of scales each unit loads in the plane's single coalesced scale read (scale
    /// vectorization).
    vectorization_scale: usize,
    /// Number of scales the plane reads, all in one coalesced load
    /// (`plane_size * vectorization_scale`).
    num_scale_per_plane: usize,
    /// Number of dequantized values the plane produces (`num_scale_per_plane * block_size`).
    num_value_per_plane: usize,
    /// Number of packed store elements the plane processes (`num_value_per_plane / num_quants`).
    num_store_per_plane: usize,
    /// Number of coalesced store-load rounds each unit performs to cover the plane's chunk
    /// (`num_store_per_plane / (plane_size * vectorization_store)`).
    num_store_loads_per_unit: usize,
}

/// Compute the per-plane dequantization blueprint.
///
/// The plane reads a single coalesced vector of `plane_size * vectorization_scale` scales. Those
/// scales define the chunk of values the plane owns (`num_scale_per_plane * block_size`), which is
/// then streamed in as coalesced rounds where every unit loads `vectorization_store` contiguous
/// packed elements. The dequantized value count must therefore decompose into whole, plane-wide
/// store rounds: `num_value_per_plane` must be a multiple of `plane_size * vectorization_store *
/// num_quants`.
///
/// The ratio is `vectorization_scale * block_size / (vectorization_store * num_quants)`. When it is
/// not a whole number we narrow the vectorization until it is, halving down to `1`:
/// - while the chunk spans at least one store round, narrow the scale read (`vectorization_scale`);
/// - if a single store round would overshoot the chunk, narrow the store read
///   (`vectorization_store`) instead.
///
/// This always terminates: a `block_size` that is a multiple of `num_quants` (required for packed
/// storage) makes the alignment exact once both factors reach `1`.
fn calculate_blueprint(
    best_vectorization_store: usize,
    best_vectorization_scale: usize,
    scheme: &QuantScheme,
    block_size: usize,
    plane_size: usize,
) -> DequantizationBlueprint {
    let num_quants = scheme.num_quants();

    let mut vectorization_scale = best_vectorization_scale;
    let mut vectorization_store = best_vectorization_store;

    loop {
        let num_value_per_plane = plane_size * vectorization_scale * block_size;
        let num_value_per_round = plane_size * vectorization_store * num_quants;

        if num_value_per_plane.is_multiple_of(num_value_per_round) {
            break;
        }

        if num_value_per_plane >= num_value_per_round && vectorization_scale > 1 {
            // Multiple store rounds but not a whole number: narrow the coalesced scale read.
            vectorization_scale /= 2;
        } else if vectorization_store > 1 {
            // A single store round overshoots the chunk: narrow the store read instead.
            vectorization_store /= 2;
        } else {
            // Cannot tile any further (block_size not a multiple of num_quants); guarded upstream.
            break;
        }
    }

    let num_scale_per_plane = plane_size * vectorization_scale;
    let num_value_per_plane = num_scale_per_plane * block_size;
    let num_store_per_plane = num_value_per_plane / num_quants;
    let num_store_loads_per_unit = num_store_per_plane / (plane_size * vectorization_store);

    DequantizationBlueprint {
        plane_size,
        vectorization_store,
        vectorization_scale,
        num_scale_per_plane,
        num_value_per_plane,
        num_store_per_plane,
        num_store_loads_per_unit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme::{QuantLevel, QuantStore, QuantValue};

    /// Build a `PackedU32` scheme for the given quantized value type. For `PackedU32`, the number of
    /// packed values is `32 / value.size_bits()` (e.g. 4 for Q8, 8 for Q4, 16 for Q2).
    fn scheme(value: QuantValue) -> QuantScheme {
        QuantScheme::default()
            .with_value(value)
            .with_store(QuantStore::PackedU32(0))
            .with_level(QuantLevel::block([1]))
    }

    /// Assert the structural invariants every blueprint must uphold, independent of the inputs.
    fn assert_blueprint_invariants(
        bp: &DequantizationBlueprint,
        scheme: &QuantScheme,
        block_size: usize,
    ) {
        let num_quants = scheme.num_quants();

        assert!(bp.plane_size >= 1);
        assert!(bp.vectorization_store >= 1);
        assert!(bp.vectorization_scale >= 1);

        // The plane reads its scales as a single coalesced vector.
        assert_eq!(
            bp.num_scale_per_plane,
            bp.plane_size * bp.vectorization_scale,
            "scales per plane must equal plane_size * scale vectorization"
        );
        // Those scales define the value chunk the plane owns.
        assert_eq!(
            bp.num_value_per_plane,
            bp.num_scale_per_plane * block_size,
            "values per plane must equal scales per plane * block size"
        );
        // The values decompose into whole packed store elements.
        assert_eq!(
            bp.num_value_per_plane,
            bp.num_store_per_plane * num_quants,
            "values per plane must equal store elements * num_quants"
        );
        // The chunk decomposes into whole, plane-wide coalesced store rounds.
        let store_per_round = bp.plane_size * bp.vectorization_store;
        assert!(
            bp.num_store_per_plane.is_multiple_of(store_per_round),
            "store elements per plane ({}) must be a multiple of a store round ({store_per_round})",
            bp.num_store_per_plane,
        );
        assert_eq!(
            bp.num_store_loads_per_unit,
            bp.num_store_per_plane / store_per_round,
            "store loads per unit must equal store rounds in the chunk"
        );
        assert!(
            bp.num_store_loads_per_unit >= 1,
            "every unit must perform at least one store load"
        );
    }

    #[test]
    fn block_aligned_with_store_round() {
        // Q4 in u32 => 8 values per store element. Plane of 32, vec4 store => 1024 values per round.
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 4, &scheme, 32, 32);

        // vec_scale * block / (vec_store * num_quants) = 4 * 32 / (4 * 8) = 4 store rounds; aligned.
        assert_eq!(bp.vectorization_store, 4);
        assert_eq!(bp.vectorization_scale, 4);
        assert_eq!(bp.num_scale_per_plane, 128);
        assert_eq!(bp.num_value_per_plane, 4096);
        assert_eq!(bp.num_store_per_plane, 512);
        assert_eq!(bp.num_store_loads_per_unit, 4);
        assert_blueprint_invariants(&bp, &scheme, 32);
    }

    #[test]
    fn already_aligned_keeps_both_vectorizations() {
        // 32 * 2 * 64 = 4096 values; round = 32 * 4 * 8 = 1024; ratio 4. Nothing to narrow.
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 2, &scheme, 64, 32);

        assert_eq!(bp.vectorization_store, 4);
        assert_eq!(bp.vectorization_scale, 2);
        assert_eq!(bp.num_store_loads_per_unit, 4);
        assert_blueprint_invariants(&bp, &scheme, 64);
    }

    #[test]
    fn store_round_overshoot_narrows_store_vectorization() {
        // block 16, q4, vec_store 4: round (1024) overshoots the chunk (32*1*16 = 512) even at
        // vec_scale 1, so the store read is narrowed instead.
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 1, &scheme, 16, 32);

        // vec_store halved 4 -> 2 so the round (512) matches the chunk (512); vec_scale untouched.
        assert_eq!(bp.vectorization_scale, 1);
        assert_eq!(bp.vectorization_store, 2);
        assert_eq!(bp.num_value_per_plane, 512);
        assert_eq!(bp.num_store_loads_per_unit, 1);
        assert_blueprint_invariants(&bp, &scheme, 16);
    }

    #[test]
    fn small_block_narrows_store_vectorization_to_one() {
        // block 8, q4, vec_store 4: round overshoots until vec_store reaches 1.
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 1, &scheme, 8, 32);

        assert_eq!(bp.vectorization_store, 1);
        assert_eq!(bp.vectorization_scale, 1);
        assert_eq!(bp.num_value_per_plane, 256);
        assert_eq!(bp.num_store_loads_per_unit, 1);
        assert_blueprint_invariants(&bp, &scheme, 8);
    }

    #[test]
    fn narrows_scale_then_store_when_needed() {
        // block 24 (= 3 * num_quants), q4, vec_store 4, vec_scale 2.
        // ratio 2*24/(4*8) = 1.5 -> chunk >= round so narrow scale: vec_scale 2 -> 1.
        // ratio 1*24/(4*8) = 0.75 -> chunk < round so narrow store: vec_store 4 -> 1 (ratio 3).
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 2, &scheme, 24, 32);

        assert_eq!(bp.vectorization_scale, 1);
        assert_eq!(bp.vectorization_store, 1);
        assert_eq!(bp.num_value_per_plane, 32 * 1 * 24);
        assert_eq!(bp.num_store_loads_per_unit, 3);
        assert_blueprint_invariants(&bp, &scheme, 24);
    }

    #[test]
    fn handles_q8_and_q2_packing() {
        // Q8 => 4 values per store element. block 8, vec_store 4, vec_scale 4, plane 32.
        // round = 32*4*4 = 512; chunk = 32*4*8 = 1024; ratio 2. Aligned.
        let scheme_q8 = scheme(QuantValue::Q8F);
        let bp = calculate_blueprint(4, 4, &scheme_q8, 8, 32);
        assert_eq!(bp.vectorization_scale, 4);
        assert_eq!(bp.vectorization_store, 4);
        assert_eq!(bp.num_store_loads_per_unit, 2);
        assert_blueprint_invariants(&bp, &scheme_q8, 8);

        // Q2 => 16 values per store element. block 32, vec_store 2, vec_scale 2, plane 32.
        // round = 32*2*16 = 1024; chunk = 32*2*32 = 2048; ratio 2. Aligned.
        let scheme_q2 = scheme(QuantValue::Q2F);
        let bp = calculate_blueprint(2, 2, &scheme_q2, 32, 32);
        assert_eq!(bp.vectorization_scale, 2);
        assert_eq!(bp.vectorization_store, 2);
        assert_eq!(bp.num_store_loads_per_unit, 2);
        assert_blueprint_invariants(&bp, &scheme_q2, 32);
    }

    #[test]
    fn single_unit_plane() {
        // A plane of one unit (e.g. CPU) still produces a valid blueprint.
        let scheme = scheme(QuantValue::Q4F);
        let bp = calculate_blueprint(4, 4, &scheme, 32, 1);

        assert_eq!(bp.plane_size, 1);
        assert_eq!(bp.num_scale_per_plane, 4);
        assert_eq!(bp.num_value_per_plane, 128);
        assert_eq!(bp.num_store_loads_per_unit, 4);
        assert_blueprint_invariants(&bp, &scheme, 32);
    }

    #[test]
    fn invariants_hold_across_many_configurations() {
        let values = [QuantValue::Q8F, QuantValue::Q4F, QuantValue::Q2F];
        let stores = [1, 2, 4, 8];
        let scales = [1, 2, 4, 8];
        let planes = [1, 16, 32, 64];

        for value in values {
            let scheme = scheme(value);
            let num_quants = scheme.num_quants();
            // Packed storage requires the block size to be a multiple of num_quants; otherwise the
            // chunk can never decompose into whole store elements. Exercise such valid blocks.
            let blocks = [1, 2, 4, 8].map(|m| m * num_quants);

            for &store in &stores {
                for &scale in &scales {
                    for &plane in &planes {
                        for &block in &blocks {
                            let bp = calculate_blueprint(store, scale, &scheme, block, plane);
                            assert_blueprint_invariants(&bp, &scheme, block);
                            // The blueprint only ever narrows the requested vectorizations.
                            assert!(bp.vectorization_scale <= scale);
                            assert!(bp.vectorization_store <= store);
                        }
                    }
                }
            }
        }
    }
}

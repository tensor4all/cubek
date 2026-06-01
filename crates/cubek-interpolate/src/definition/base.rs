use crate::{
    components::readers::ReaderType,
    definition::{Bicubic, Bilinear, InterpolatePrecision, Lanczos3, Nearest},
    routines::InterpolateBlueprint,
};
use cubecl::prelude::*;

// Base trait for interpolation algorithms.
#[cube]
pub trait Interpolate {
    const HALO: usize;

    fn compute_weight<EA: Float>(x: EA) -> EA;

    fn compute_value<P: InterpolatePrecision, N: Size>(
        input: &Tensor<Vector<P::EI, N>>,
        input_height: usize,
        input_width: usize,
        base_row: isize,
        base_col: isize,
        frac_row: P::EA,
        frac_col: P::EA,
        reader: ReaderType<P::EA, N>,
        #[comptime] blueprint: InterpolateBlueprint,
    ) -> Vector<P::EI, N>;
}

/// Algorithm used for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum InterpolateMode {
    /// Nearest-neighbor interpolation.
    /// <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
    Nearest(NearestMode),

    /// Bilinear interpolation.
    /// <https://en.wikipedia.org/wiki/Bilinear_interpolation>
    Bilinear,

    /// Bicubic interpolation.
    /// <https://en.wikipedia.org/wiki/Bicubic_interpolation>
    Bicubic,

    /// Lanczos3 interpolation (6-tap sinc-based filter).
    /// <https://en.wikipedia.org/wiki/Lanczos_resampling>
    Lanczos3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum NearestMode {
    // Matches Scikit-Image and PIL nearest neighbours interpolation algorithms.
    Exact,
    // Matches buggy OpenCV’s INTER_NEAREST interpolation algorithm for backward compatibility.
    Floor,
}

/// Interpolation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateOptions {
    /// Algorithm used.
    pub mode: InterpolateMode,
    /// If `true`, the input and output tensors are aligned by their corner pixels.
    /// If `false`, half-pixel coordinate mapping is used instead.
    pub align_corners: bool,
}

impl InterpolateOptions {
    /// Create new interpolate options with the given mode.
    /// Defaults to `align_corners = true`.
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            align_corners: true,
        }
    }

    /// Set align_corners.
    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

// Helper functions to map InterpolateMode to the corresponding Interpolate implementation.
pub fn get_halo(mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest(_) => <Nearest as Interpolate>::HALO,
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::HALO,
        InterpolateMode::Bicubic => <Bicubic as Interpolate>::HALO,
        InterpolateMode::Lanczos3 => <Lanczos3 as Interpolate>::HALO,
    }
}

// Maps InterpolateMode to the corresponding Interpolate implementation for compute_value.
#[cube]
pub fn compute_value<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_height: usize,
    input_width: usize,
    base_row: isize,
    base_col: isize,
    frac_row: P::EA,
    frac_col: P::EA,
    reader: ReaderType<P::EA, N>,
    #[comptime] blueprint: InterpolateBlueprint,
) -> Vector<P::EI, N> {
    match blueprint.options.mode {
        InterpolateMode::Nearest(_) => Nearest::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
            blueprint,
        ),
        InterpolateMode::Bilinear => Bilinear::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
            blueprint,
        ),
        InterpolateMode::Bicubic => Bicubic::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
            blueprint,
        ),
        InterpolateMode::Lanczos3 => Lanczos3::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
            blueprint,
        ),
    }
}

#[cube]
pub fn compute_value_default<I: Interpolate, P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_height: usize,
    input_width: usize,
    base_row: isize,
    base_col: isize,
    frac_row: P::EA,
    frac_col: P::EA,
    reader: ReaderType<P::EA, N>,
    #[comptime] blueprint: InterpolateBlueprint,
) -> Vector<P::EI, N> {
    let halo = comptime!(I::HALO);
    let radius_offset = ((halo - 1) / 2) as isize;

    let mut weights_col = Array::new(halo);
    #[unroll]
    for j in 0..halo {
        let d_col = P::EA::cast_from(j as isize - radius_offset) - frac_col;
        weights_col[j] = Vector::cast_from(I::compute_weight::<P::EA>(d_col));
    }

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::<P::EA, N>::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::<P::EA, N>::zeroed();

        let row = base_row + i as isize - radius_offset;

        #[unroll]
        for j in 0..halo {
            let col = base_col + j as isize - radius_offset;

            let is_in_bounds = is_in_bounds(col, input_width, blueprint.options)
                && is_in_bounds(row, input_height, blueprint.options);

            let clamped_row = row.max(0).min(input_height as isize - 1) as usize;
            let clamped_col = col.max(0).min(input_width as isize - 1) as usize;

            let weight_col = weights_col[j];

            row_value += select(
                is_in_bounds,
                reader.read_weighted::<P::EI>(input, clamped_row, clamped_col, weight_col),
                Vector::zeroed(),
            );
            row_weight_sum += select(
                is_in_bounds,
                Vector::cast_from(weight_col),
                Vector::zeroed(),
            );
        }

        let d_row = P::EA::cast_from(i as isize - radius_offset) - frac_row;
        let weight_row = Vector::<P::EA, N>::cast_from(I::compute_weight::<P::EA>(d_row));

        final_value += row_value * Vector::cast_from(weight_row);
        total_weight += row_weight_sum * Vector::cast_from(weight_row);
    }

    let epsilon = Vector::<P::EA, N>::cast_from(P::EA::new(1e-7));

    Vector::cast_from(final_value / total_weight.max(epsilon))
}

// Only used for bounds checking in Lanczos3 mode.
#[cube]
fn is_in_bounds(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Lanczos3 => value >= 0 && value < size as isize,
        _ => true,
    }
}

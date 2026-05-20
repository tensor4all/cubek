use cubek_std::MatrixLayout;

use crate::components::{
    batch::{BatchConfig, CheckBounds},
    global::memory::GlobalLayoutConfig,
};

/// Config for the Mosaic family. Intentionally small — only the knobs the
/// kernel currently consumes. The accumulator choice (eventually carried
/// in here) is what should drive the rest of the algorithm.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MosaicConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for MosaicConfig {
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
}
